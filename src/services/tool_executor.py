"""Tool executor for local AI function calling.

Provides tools that local AI models (Ollama) can call to:
- Access filesystem (read files, search code)
- Query hafs context (embeddings, knowledge graph, scratchpad)
- Execute safe commands
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes tools/functions for local AI models.

    Provides safe, sandboxed tool execution with access to:
    - File operations
    - Code search
    - hafs context system
    - Safe command execution
    """

    def __init__(
        self,
        context_root: Optional[Path] = None,
        allowed_commands: Optional[list[str]] = None,
    ):
        """Initialize tool executor.

        Args:
            context_root: Root directory for hafs context (default: ~/.context)
            allowed_commands: Whitelist of allowed shell commands
        """
        self.context_root = context_root or Path.home() / ".context"
        self.allowed_commands = allowed_commands or [
            "git",
            "ls",
            "find",
            "grep",
            "cat",
            "hafs",
            "rg",
            "fd",
        ]

        # Register available tools
        self.tools = {
            "read_file": self._read_file,
            "list_directory": self._list_directory,
            "search_code": self._search_code,
            "find_files": self._find_files,
            "read_scratchpad": self._read_scratchpad,
            "query_embeddings": self._query_embeddings,
            "run_command": self._run_command,
            "get_file_info": self._get_file_info,
        }

    async def execute(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call from LLM.

        Args:
            tool_call: Tool call dict with 'name' and 'parameters'

        Returns:
            Result dict with 'result' or 'error'
        """
        tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name")
        parameters = tool_call.get("parameters") or tool_call.get(
            "function", {}
        ).get("arguments", {})

        if not tool_name:
            return {"error": "No tool name provided"}

        if tool_name not in self.tools:
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys()),
            }

        try:
            logger.info(f"Executing tool: {tool_name} with params: {parameters}")
            result = await self.tools[tool_name](**parameters)
            return {"result": result, "tool_name": tool_name}
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"error": str(e), "tool_name": tool_name}

    async def _read_file(self, path: str, max_lines: int = 1000) -> str:
        """Read file contents.

        Args:
            path: File path (relative or absolute)
            max_lines: Maximum lines to read

        Returns:
            File contents (truncated if > max_lines)
        """
        file_path = Path(path).expanduser().resolve()

        # Security check: prevent reading outside allowed directories
        if not self._is_safe_path(file_path):
            raise PermissionError(
                f"Access denied: {path} (outside allowed directories)"
            )

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")

        # Read file
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()

            if len(lines) > max_lines:
                return (
                    "".join(lines[:max_lines])
                    + f"\n\n... (truncated, {len(lines)} total lines)"
                )

            return "".join(lines)

        except UnicodeDecodeError:
            # Try binary read for non-text files
            with open(file_path, "rb") as f:
                data = f.read(4096)  # First 4KB
                return f"Binary file (showing first 4KB):\n{data.hex()}"

    async def _list_directory(
        self, path: str = ".", max_items: int = 100
    ) -> list[dict[str, Any]]:
        """List directory contents.

        Args:
            path: Directory path
            max_items: Maximum items to return

        Returns:
            List of file/directory info dicts
        """
        dir_path = Path(path).expanduser().resolve()

        if not self._is_safe_path(dir_path):
            raise PermissionError(f"Access denied: {path}")

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        items = []
        for item in sorted(dir_path.iterdir())[:max_items]:
            try:
                stat = item.stat()
                items.append(
                    {
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size_bytes": stat.st_size,
                        "modified": stat.st_mtime,
                    }
                )
            except (OSError, PermissionError):
                continue

        if len(list(dir_path.iterdir())) > max_items:
            items.append(
                {"name": f"... ({len(list(dir_path.iterdir()))} total items)"}
            )

        return items

    async def _search_code(
        self, pattern: str, directory: str = ".", max_results: int = 50
    ) -> list[dict[str, Any]]:
        """Search code using ripgrep.

        Args:
            pattern: Regex pattern to search
            directory: Directory to search
            max_results: Maximum results

        Returns:
            List of match dicts
        """
        dir_path = Path(directory).expanduser().resolve()

        if not self._is_safe_path(dir_path):
            raise PermissionError(f"Access denied: {directory}")

        # Use ripgrep for fast search
        cmd = [
            "rg",
            "--json",
            "--max-count",
            str(max_results),
            pattern,
            str(dir_path),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            results = []
            for line in stdout.decode().strip().split("\n"):
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if data.get("type") == "match":
                        results.append(
                            {
                                "file": data["data"]["path"]["text"],
                                "line": data["data"]["line_number"],
                                "text": data["data"]["lines"]["text"].strip(),
                            }
                        )
                except json.JSONDecodeError:
                    continue

            return results[:max_results]

        except FileNotFoundError:
            # Fallback to grep if ripgrep not available
            return await self._search_code_fallback(pattern, dir_path, max_results)

    async def _search_code_fallback(
        self, pattern: str, directory: Path, max_results: int
    ) -> list[dict[str, Any]]:
        """Fallback search using grep."""
        cmd = ["grep", "-r", "-n", pattern, str(directory)]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            results = []
            for line in stdout.decode().strip().split("\n")[:max_results]:
                if not line:
                    continue

                # Parse grep output: file:line:text
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    results.append(
                        {
                            "file": parts[0],
                            "line": int(parts[1]) if parts[1].isdigit() else 0,
                            "text": parts[2].strip(),
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Search fallback failed: {e}")
            return []

    async def _find_files(
        self, name_pattern: str, directory: str = ".", max_results: int = 100
    ) -> list[str]:
        """Find files matching name pattern.

        Args:
            name_pattern: Glob pattern (e.g., "*.py")
            directory: Directory to search
            max_results: Maximum results

        Returns:
            List of matching file paths
        """
        dir_path = Path(directory).expanduser().resolve()

        if not self._is_safe_path(dir_path):
            raise PermissionError(f"Access denied: {directory}")

        # Use glob for simple patterns
        results = []
        for match in dir_path.rglob(name_pattern):
            if match.is_file():
                results.append(str(match))
                if len(results) >= max_results:
                    break

        return results

    async def _read_scratchpad(
        self, category: str = "state"
    ) -> dict[str, Any]:
        """Read hafs scratchpad.

        Args:
            category: Scratchpad category (state, metacognition, epistemic)

        Returns:
            Scratchpad contents
        """
        scratchpad_file = self.context_root / "scratchpad" / f"{category}.json"

        if not scratchpad_file.exists():
            return {"error": f"Scratchpad not found: {category}"}

        try:
            with open(scratchpad_file) as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to read scratchpad: {e}"}

    async def _query_embeddings(
        self,
        query: str,
        context_id: str = "default",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Query hafs embeddings.

        Args:
            query: Search query
            context_id: Context/domain ID
            top_k: Number of results

        Returns:
            Search results
        """
        # Check if embedding service is available
        embedding_index = (
            self.context_root / "knowledge" / context_id / "embeddings" / "index.json"
        )

        if not embedding_index.exists():
            return [{"error": f"No embeddings found for context: {context_id}"}]

        try:
            # Load index
            with open(embedding_index) as f:
                index = json.load(f)

            # Simple keyword search for now
            # TODO: Replace with actual vector search when embedding service is available
            results = []
            query_terms = query.lower().split()

            for emb_id, emb_file in list(index.items())[:top_k]:
                emb_path = embedding_index.parent / emb_file

                if emb_path.exists():
                    try:
                        with open(emb_path) as f:
                            emb_data = json.load(f)

                            # Simple relevance score based on keyword overlap
                            text = emb_data.get("text", "").lower()
                            score = sum(1 for term in query_terms if term in text)

                            if score > 0:
                                results.append(
                                    {
                                        "id": emb_id,
                                        "title": emb_data.get("title", "Unknown"),
                                        "snippet": text[:200],
                                        "score": score,
                                    }
                                )
                    except Exception:
                        continue

            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        except Exception as e:
            return [{"error": f"Failed to query embeddings: {e}"}]

    async def _run_command(
        self, command: str, cwd: Optional[str] = None
    ) -> str:
        """Run safe shell command.

        Args:
            command: Command to run
            cwd: Working directory

        Returns:
            Command output
        """
        # Security: whitelist of allowed commands
        cmd_name = command.split()[0]

        if cmd_name not in self.allowed_commands:
            raise PermissionError(
                f"Command not allowed: {cmd_name}. "
                f"Allowed: {', '.join(self.allowed_commands)}"
            )

        # Resolve working directory
        if cwd:
            cwd_path = Path(cwd).expanduser().resolve()
            if not self._is_safe_path(cwd_path):
                raise PermissionError(f"Access denied: {cwd}")
            cwd = str(cwd_path)

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
            )

            if result.returncode != 0:
                return f"Error (exit {result.returncode}): {result.stderr}"

            return result.stdout

        except subprocess.TimeoutExpired:
            return "Error: Command timeout (30 seconds)"
        except Exception as e:
            return f"Error executing command: {e}"

    async def _get_file_info(self, path: str) -> dict[str, Any]:
        """Get file metadata.

        Args:
            path: File path

        Returns:
            File info dict
        """
        file_path = Path(path).expanduser().resolve()

        if not self._is_safe_path(file_path):
            raise PermissionError(f"Access denied: {path}")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        stat = file_path.stat()

        return {
            "path": str(file_path),
            "name": file_path.name,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024**2),
            "modified": stat.st_mtime,
            "is_directory": file_path.is_dir(),
            "is_file": file_path.is_file(),
            "extension": file_path.suffix,
        }

    def _is_safe_path(self, path: Path) -> bool:
        """Check if path is safe to access.

        Args:
            path: Path to check

        Returns:
            True if safe
        """
        # Allow access to:
        # - Current working directory
        # - Home directory
        # - hafs context directory
        # - Common project directories

        allowed_roots = [
            Path.cwd(),
            Path.home(),
            self.context_root,
            Path.home() / "Code",
            Path.home() / "projects",
        ]

        # Check if path is under allowed root
        try:
            resolved = path.resolve()
            for root in allowed_roots:
                try:
                    resolved.relative_to(root.resolve())
                    return True
                except ValueError:
                    continue
            return False
        except Exception:
            return False


# Tool definitions for Ollama function calling
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a text file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file (relative or absolute)",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum lines to read (default: 1000)",
                        "default": 1000,
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search code using regex pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search (default: current dir)",
                        "default": ".",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (default: 50)",
                        "default": 50,
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Find files matching name pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '*.py', '*.json')",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search",
                        "default": ".",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 100,
                    },
                },
                "required": ["name_pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List directory contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path",
                        "default": ".",
                    },
                    "max_items": {
                        "type": "integer",
                        "description": "Maximum items to return",
                        "default": 100,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_scratchpad",
            "description": "Read hafs scratchpad state",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Scratchpad category",
                        "enum": ["state", "metacognition", "epistemic"],
                        "default": "state",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_embeddings",
            "description": "Search hafs knowledge embeddings",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "context_id": {
                        "type": "string",
                        "description": "Context ID (e.g., 'filesystem', 'code')",
                        "default": "default",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_info",
            "description": "Get file metadata (size, modified time, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path",
                    },
                },
                "required": ["path"],
            },
        },
    },
]
