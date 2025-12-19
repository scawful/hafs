"""Generalized Code Description Generator.

Provides language-agnostic routine/function description generation
with plugin support for multiple languages.

Supported languages:
- Assembly (65816, Z80, x86)
- C/C++
- Python
- Java
- Lisp/Scheme
- Rust
- Go

Usage:
    describer = CodeDescriber()
    await describer.setup()

    # Describe a single function
    desc = await describer.describe_function(code, language="cpp")

    # Batch describe from files
    results = await describer.describe_project(
        path="/path/to/project",
        language="cpp",
        limit=100
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from hafs.agents.base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class CodeUnit:
    """A unit of code (function, method, class, etc.)."""

    name: str
    kind: str  # function, method, class, macro, etc.
    language: str
    code: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    signature: Optional[str] = None
    description: Optional[str] = None
    calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LanguageConfig:
    """Configuration for a programming language."""

    name: str
    extensions: List[str]
    comment_patterns: List[str]
    function_patterns: List[re.Pattern]
    class_patterns: List[re.Pattern] = field(default_factory=list)
    prompt_template: str = ""
    context_hints: List[str] = field(default_factory=list)


class LanguagePlugin(ABC):
    """Base class for language-specific plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Language name."""
        pass

    @property
    @abstractmethod
    def extensions(self) -> List[str]:
        """File extensions for this language."""
        pass

    @abstractmethod
    def extract_units(self, code: str, file_path: Optional[str] = None) -> List[CodeUnit]:
        """Extract code units (functions, classes, etc.) from source code."""
        pass

    @abstractmethod
    def build_prompt(self, unit: CodeUnit) -> str:
        """Build the LLM prompt for describing this code unit."""
        pass

    def get_context_hints(self) -> List[str]:
        """Return context hints for better descriptions."""
        return []


class AssemblyPlugin(LanguagePlugin):
    """Plugin for assembly languages (65816, Z80, x86)."""

    name = "assembly"
    extensions = [".asm", ".s", ".inc", ".a65", ".z80"]

    # Patterns for different assembly dialects
    LABEL_PATTERNS = [
        re.compile(r'^([A-Za-z_][A-Za-z0-9_]*):'),  # Standard label:
        re.compile(r'^\.?([A-Za-z_][A-Za-z0-9_]*)\s*='),  # Constant assignment
        re.compile(r'^([A-Za-z_][A-Za-z0-9_]*):\s*;'),  # Label with comment
    ]

    ROUTINE_INDICATORS = ["RTS", "RTL", "RET", "RETF", "ret"]

    def extract_units(self, code: str, file_path: Optional[str] = None) -> List[CodeUnit]:
        units = []
        lines = code.split('\n')
        current_label = None
        current_code = []
        current_line = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Check for label
            for pattern in self.LABEL_PATTERNS:
                match = pattern.match(stripped)
                if match:
                    # Save previous routine
                    if current_label and current_code:
                        units.append(CodeUnit(
                            name=current_label,
                            kind="routine",
                            language="assembly",
                            code='\n'.join(current_code),
                            file_path=file_path,
                            line_number=current_line,
                        ))

                    current_label = match.group(1)
                    current_code = [line]
                    current_line = i + 1
                    break
            else:
                if current_label:
                    current_code.append(line)

                    # Check for routine end
                    for indicator in self.ROUTINE_INDICATORS:
                        if indicator in stripped.upper():
                            units.append(CodeUnit(
                                name=current_label,
                                kind="routine",
                                language="assembly",
                                code='\n'.join(current_code),
                                file_path=file_path,
                                line_number=current_line,
                            ))
                            current_label = None
                            current_code = []
                            break

        # Don't forget last routine
        if current_label and current_code:
            units.append(CodeUnit(
                name=current_label,
                kind="routine",
                language="assembly",
                code='\n'.join(current_code),
                file_path=file_path,
                line_number=current_line,
            ))

        return units

    def build_prompt(self, unit: CodeUnit) -> str:
        code_snippet = unit.code[:800] if len(unit.code) > 800 else unit.code

        return f"""Analyze this assembly routine and provide a concise 1-2 sentence description of its purpose.

Routine: {unit.name}
File: {unit.file_path or 'Unknown'}

Code:
{code_snippet}

Respond with ONLY the description. Focus on what the routine does functionally (data processing, I/O, graphics, audio, game logic, etc.)."""

    def get_context_hints(self) -> List[str]:
        return [
            "65816 is a 16-bit processor used in SNES",
            "Common patterns: JSR/JSL for calls, RTS/RTL for returns",
            "LDA/STA for load/store, BRA/BEQ/BNE for branches",
        ]


class CppPlugin(LanguagePlugin):
    """Plugin for C/C++ code."""

    name = "cpp"
    extensions = [".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"]

    # Function patterns
    FUNCTION_PATTERN = re.compile(
        r'^(?:static\s+|inline\s+|virtual\s+|extern\s+)*'
        r'(?:const\s+)?'
        r'(\w+(?:\s*[*&]+\s*|\s+))'  # Return type
        r'(\w+)\s*'  # Function name
        r'\(([^)]*)\)\s*'  # Parameters
        r'(?:const\s*)?'
        r'(?:override\s*)?'
        r'(?:noexcept\s*)?'
        r'\{',
        re.MULTILINE
    )

    # Method pattern (inside class)
    METHOD_PATTERN = re.compile(
        r'(\w+)::(\w+)\s*\(([^)]*)\)',
        re.MULTILINE
    )

    # Class pattern
    CLASS_PATTERN = re.compile(
        r'^(?:template\s*<[^>]*>\s*)?'
        r'(?:class|struct)\s+(\w+)',
        re.MULTILINE
    )

    def extract_units(self, code: str, file_path: Optional[str] = None) -> List[CodeUnit]:
        units = []

        # Extract functions
        for match in self.FUNCTION_PATTERN.finditer(code):
            return_type = match.group(1).strip()
            func_name = match.group(2)
            params = match.group(3)

            # Find function body
            start = match.end() - 1  # Start at opening brace
            body = self._extract_braced_block(code, start)

            units.append(CodeUnit(
                name=func_name,
                kind="function",
                language="cpp",
                code=body[:1000],
                file_path=file_path,
                line_number=code[:match.start()].count('\n') + 1,
                signature=f"{return_type} {func_name}({params})",
            ))

        # Extract class methods
        for match in self.METHOD_PATTERN.finditer(code):
            class_name = match.group(1)
            method_name = match.group(2)
            params = match.group(3)

            units.append(CodeUnit(
                name=f"{class_name}::{method_name}",
                kind="method",
                language="cpp",
                code=code[match.start():match.start()+500],
                file_path=file_path,
                line_number=code[:match.start()].count('\n') + 1,
                signature=f"{class_name}::{method_name}({params})",
                metadata={"class": class_name},
            ))

        return units

    def _extract_braced_block(self, code: str, start: int) -> str:
        """Extract a braced block starting at position."""
        if start >= len(code) or code[start] != '{':
            return ""

        depth = 0
        end = start

        for i in range(start, min(start + 2000, len(code))):
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        return code[start:end]

    def build_prompt(self, unit: CodeUnit) -> str:
        code_snippet = unit.code[:800] if len(unit.code) > 800 else unit.code

        return f"""Analyze this C/C++ {unit.kind} and provide a concise 1-2 sentence description of its purpose.

{unit.kind.title()}: {unit.name}
Signature: {unit.signature or 'N/A'}
File: {unit.file_path or 'Unknown'}

Code:
{code_snippet}

Respond with ONLY the description. Focus on what the {unit.kind} does, its inputs/outputs, and any side effects."""

    def get_context_hints(self) -> List[str]:
        return [
            "Look for memory management (malloc/free, new/delete)",
            "Identify pointer operations and references",
            "Note any hardware interaction or system calls",
        ]


class PythonPlugin(LanguagePlugin):
    """Plugin for Python code."""

    name = "python"
    extensions = [".py", ".pyw", ".pyi"]

    FUNCTION_PATTERN = re.compile(
        r'^(\s*)(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?\s*:',
        re.MULTILINE
    )

    CLASS_PATTERN = re.compile(
        r'^(\s*)class\s+(\w+)(?:\(([^)]*)\))?\s*:',
        re.MULTILINE
    )

    def extract_units(self, code: str, file_path: Optional[str] = None) -> List[CodeUnit]:
        units = []
        lines = code.split('\n')

        for match in self.FUNCTION_PATTERN.finditer(code):
            indent = len(match.group(1))
            func_name = match.group(2)
            params = match.group(3)
            return_type = match.group(4)

            # Extract function body based on indentation
            start_line = code[:match.start()].count('\n')
            body_lines = [lines[start_line]]

            for i in range(start_line + 1, min(start_line + 50, len(lines))):
                line = lines[i]
                if line.strip() and not line.startswith(' ' * (indent + 1)) and not line.startswith('\t' * (indent // 4 + 1)):
                    if not line.strip().startswith('#'):
                        break
                body_lines.append(line)

            # Check for docstring
            docstring = None
            body_code = '\n'.join(body_lines)
            doc_match = re.search(r'"""(.+?)"""', body_code, re.DOTALL)
            if doc_match:
                docstring = doc_match.group(1).strip()

            units.append(CodeUnit(
                name=func_name,
                kind="function",
                language="python",
                code=body_code[:1000],
                file_path=file_path,
                line_number=start_line + 1,
                signature=f"def {func_name}({params})" + (f" -> {return_type}" if return_type else ""),
                description=docstring,
            ))

        return units

    def build_prompt(self, unit: CodeUnit) -> str:
        # If there's already a docstring, we might want to enhance it
        existing = f"\nExisting docstring: {unit.description}" if unit.description else ""
        code_snippet = unit.code[:800] if len(unit.code) > 800 else unit.code

        return f"""Analyze this Python function and provide a concise 1-2 sentence description of its purpose.

Function: {unit.name}
Signature: {unit.signature or 'N/A'}
File: {unit.file_path or 'Unknown'}{existing}

Code:
{code_snippet}

Respond with ONLY the description. Focus on what the function does, its parameters, return value, and any important behavior."""

    def get_context_hints(self) -> List[str]:
        return [
            "Look for type hints for better understanding",
            "Check for async/await patterns",
            "Note decorator usage (@property, @classmethod, etc.)",
        ]


class LispPlugin(LanguagePlugin):
    """Plugin for Lisp/Scheme code."""

    name = "lisp"
    extensions = [".lisp", ".lsp", ".cl", ".scm", ".ss", ".rkt"]

    # defun, defn, define patterns
    FUNCTION_PATTERNS = [
        re.compile(r'\(defun\s+(\S+)\s+\(([^)]*)\)', re.MULTILINE),
        re.compile(r'\(defn\s+(\S+)\s+\[([^\]]*)\]', re.MULTILINE),  # Clojure
        re.compile(r'\(define\s+\((\S+)\s+([^)]*)\)', re.MULTILINE),  # Scheme
        re.compile(r'\(define\s+(\S+)\s+\(lambda', re.MULTILINE),  # Scheme lambda
    ]

    def extract_units(self, code: str, file_path: Optional[str] = None) -> List[CodeUnit]:
        units = []

        for pattern in self.FUNCTION_PATTERNS:
            for match in pattern.finditer(code):
                func_name = match.group(1)
                params = match.group(2) if len(match.groups()) > 1 else ""

                # Extract the full s-expression
                start = match.start()
                body = self._extract_sexp(code, start)

                units.append(CodeUnit(
                    name=func_name,
                    kind="function",
                    language="lisp",
                    code=body[:1000],
                    file_path=file_path,
                    line_number=code[:start].count('\n') + 1,
                    signature=f"({func_name} {params})",
                ))

        return units

    def _extract_sexp(self, code: str, start: int) -> str:
        """Extract a complete s-expression."""
        depth = 0
        end = start

        for i in range(start, min(start + 2000, len(code))):
            if code[i] == '(':
                depth += 1
            elif code[i] == ')':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        return code[start:end]

    def build_prompt(self, unit: CodeUnit) -> str:
        code_snippet = unit.code[:800] if len(unit.code) > 800 else unit.code

        return f"""Analyze this Lisp/Scheme function and provide a concise 1-2 sentence description of its purpose.

Function: {unit.name}
Signature: {unit.signature or 'N/A'}
File: {unit.file_path or 'Unknown'}

Code:
{code_snippet}

Respond with ONLY the description. Focus on what the function computes or performs, its parameters, and return value."""


class JavaPlugin(LanguagePlugin):
    """Plugin for Java code."""

    name = "java"
    extensions = [".java"]

    METHOD_PATTERN = re.compile(
        r'(?:public|private|protected)?\s*'
        r'(?:static\s+)?'
        r'(?:final\s+)?'
        r'(?:synchronized\s+)?'
        r'(\w+(?:<[^>]+>)?)\s+'  # Return type
        r'(\w+)\s*'  # Method name
        r'\(([^)]*)\)\s*'  # Parameters
        r'(?:throws\s+[\w,\s]+)?\s*'
        r'\{',
        re.MULTILINE
    )

    CLASS_PATTERN = re.compile(
        r'(?:public\s+)?(?:abstract\s+)?(?:final\s+)?'
        r'class\s+(\w+)(?:<[^>]+>)?'
        r'(?:\s+extends\s+\w+)?'
        r'(?:\s+implements\s+[\w,\s]+)?',
        re.MULTILINE
    )

    def extract_units(self, code: str, file_path: Optional[str] = None) -> List[CodeUnit]:
        units = []

        for match in self.METHOD_PATTERN.finditer(code):
            return_type = match.group(1)
            method_name = match.group(2)
            params = match.group(3)

            # Skip constructors (return type == method name)
            if return_type == method_name:
                continue

            start = match.end() - 1
            body = self._extract_braced_block(code, start)

            units.append(CodeUnit(
                name=method_name,
                kind="method",
                language="java",
                code=body[:1000],
                file_path=file_path,
                line_number=code[:match.start()].count('\n') + 1,
                signature=f"{return_type} {method_name}({params})",
            ))

        return units

    def _extract_braced_block(self, code: str, start: int) -> str:
        if start >= len(code) or code[start] != '{':
            return ""

        depth = 0
        for i in range(start, min(start + 2000, len(code))):
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    return code[start:i + 1]

        return code[start:start + 1000]

    def build_prompt(self, unit: CodeUnit) -> str:
        code_snippet = unit.code[:800] if len(unit.code) > 800 else unit.code

        return f"""Analyze this Java method and provide a concise 1-2 sentence description of its purpose.

Method: {unit.name}
Signature: {unit.signature or 'N/A'}
File: {unit.file_path or 'Unknown'}

Code:
{code_snippet}

Respond with ONLY the description. Focus on what the method does, its parameters, return value, and any exceptions."""


class RustPlugin(LanguagePlugin):
    """Plugin for Rust code."""

    name = "rust"
    extensions = [".rs"]

    FUNCTION_PATTERN = re.compile(
        r'(?:pub\s+)?'
        r'(?:async\s+)?'
        r'fn\s+(\w+)\s*'
        r'(?:<[^>]+>)?\s*'  # Generics
        r'\(([^)]*)\)\s*'
        r'(?:->\s*([^\{]+))?\s*'
        r'\{',
        re.MULTILINE
    )

    def extract_units(self, code: str, file_path: Optional[str] = None) -> List[CodeUnit]:
        units = []

        for match in self.FUNCTION_PATTERN.finditer(code):
            func_name = match.group(1)
            params = match.group(2)
            return_type = match.group(3)

            start = match.end() - 1
            body = self._extract_braced_block(code, start)

            sig = f"fn {func_name}({params})"
            if return_type:
                sig += f" -> {return_type.strip()}"

            units.append(CodeUnit(
                name=func_name,
                kind="function",
                language="rust",
                code=body[:1000],
                file_path=file_path,
                line_number=code[:match.start()].count('\n') + 1,
                signature=sig,
            ))

        return units

    def _extract_braced_block(self, code: str, start: int) -> str:
        if start >= len(code) or code[start] != '{':
            return ""

        depth = 0
        for i in range(start, min(start + 2000, len(code))):
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    return code[start:i + 1]

        return code[start:start + 1000]

    def build_prompt(self, unit: CodeUnit) -> str:
        code_snippet = unit.code[:800] if len(unit.code) > 800 else unit.code

        return f"""Analyze this Rust function and provide a concise 1-2 sentence description of its purpose.

Function: {unit.name}
Signature: {unit.signature or 'N/A'}
File: {unit.file_path or 'Unknown'}

Code:
{code_snippet}

Respond with ONLY the description. Focus on what the function does, ownership/borrowing patterns, and any unsafe blocks."""


# Plugin registry
LANGUAGE_PLUGINS: Dict[str, Type[LanguagePlugin]] = {
    "assembly": AssemblyPlugin,
    "asm": AssemblyPlugin,
    "65816": AssemblyPlugin,
    "cpp": CppPlugin,
    "c++": CppPlugin,
    "c": CppPlugin,
    "python": PythonPlugin,
    "py": PythonPlugin,
    "lisp": LispPlugin,
    "scheme": LispPlugin,
    "clojure": LispPlugin,
    "java": JavaPlugin,
    "rust": RustPlugin,
    "rs": RustPlugin,
}


def register_plugin(name: str, plugin_class: Type[LanguagePlugin]) -> None:
    """Register a new language plugin."""
    LANGUAGE_PLUGINS[name.lower()] = plugin_class


def get_plugin(language: str) -> Optional[LanguagePlugin]:
    """Get a plugin instance for a language."""
    plugin_class = LANGUAGE_PLUGINS.get(language.lower())
    if plugin_class:
        return plugin_class()
    return None


def detect_language(file_path: str) -> Optional[str]:
    """Detect language from file extension."""
    ext = Path(file_path).suffix.lower()

    for name, plugin_class in LANGUAGE_PLUGINS.items():
        plugin = plugin_class()
        if ext in plugin.extensions:
            return name

    return None


@dataclass
class CodeKnowledgeBase:
    """Knowledge base for code descriptions with embeddings."""

    name: str
    units: List[CodeUnit] = field(default_factory=list)
    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save knowledge base to disk."""
        data = {
            "name": self.name,
            "units": [
                {
                    "name": u.name,
                    "kind": u.kind,
                    "language": u.language,
                    "code": u.code[:500],  # Truncate for storage
                    "file_path": u.file_path,
                    "line_number": u.line_number,
                    "signature": u.signature,
                    "description": u.description,
                    "calls": u.calls,
                    "called_by": u.called_by,
                    "metadata": u.metadata,
                }
                for u in self.units
            ],
            "metadata": self.metadata,
        }

        # Save units
        units_file = path / f"{self.name}_units.json"
        units_file.write_text(json.dumps(data, indent=2))

        # Save embeddings separately (they're large)
        if self.embeddings:
            embeddings_file = path / f"{self.name}_embeddings.json"
            embeddings_file.write_text(json.dumps(self.embeddings))

    @classmethod
    def load(cls, path: Path, name: str) -> "CodeKnowledgeBase":
        """Load knowledge base from disk."""
        kb = cls(name=name)

        units_file = path / f"{name}_units.json"
        if units_file.exists():
            data = json.loads(units_file.read_text())
            kb.units = [
                CodeUnit(
                    name=u["name"],
                    kind=u["kind"],
                    language=u["language"],
                    code=u.get("code", ""),
                    file_path=u.get("file_path"),
                    line_number=u.get("line_number"),
                    signature=u.get("signature"),
                    description=u.get("description"),
                    calls=u.get("calls", []),
                    called_by=u.get("called_by", []),
                    metadata=u.get("metadata", {}),
                )
                for u in data.get("units", [])
            ]
            kb.metadata = data.get("metadata", {})

        embeddings_file = path / f"{name}_embeddings.json"
        if embeddings_file.exists():
            kb.embeddings = json.loads(embeddings_file.read_text())

        return kb

    def search(
        self,
        query_embedding: List[float],
        limit: int = 20,
        min_score: float = 0.5,
    ) -> List[tuple[CodeUnit, float]]:
        """Search for similar code units using embedding similarity."""
        import numpy as np

        if not self.embeddings:
            return []

        query = np.array(query_embedding)
        results = []

        for unit in self.units:
            key = f"{unit.file_path}:{unit.name}" if unit.file_path else unit.name
            if key not in self.embeddings:
                continue

            emb = np.array(self.embeddings[key])

            # Cosine similarity
            similarity = np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb) + 1e-8)

            if similarity >= min_score:
                results.append((unit, float(similarity)))

        results.sort(key=lambda x: -x[1])
        return results[:limit]


class CodeDescriber(BaseAgent):
    """Multi-language code description generator with embedding support."""

    def __init__(self, project: str = "generic"):
        super().__init__(
            name="code_describer",
            role_description="Multi-language code description generator with embedding support"
        )
        self._project = project
        self._plugins: Dict[str, LanguagePlugin] = {}
        self._output_dir = Path.home() / ".context" / "code_descriptions"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._knowledge_bases: Dict[str, CodeKnowledgeBase] = {}

    async def setup(self) -> None:
        """Initialize the describer."""
        await super().setup()

        # Pre-instantiate common plugins
        for name in ["assembly", "cpp", "python", "java", "lisp", "rust"]:
            self._plugins[name] = get_plugin(name)

        # Load existing knowledge bases
        for kb_file in self._output_dir.glob("*_units.json"):
            name = kb_file.stem.replace("_units", "")
            try:
                self._knowledge_bases[name] = CodeKnowledgeBase.load(self._output_dir, name)
                logger.info(f"Loaded KB: {name} ({len(self._knowledge_bases[name].units)} units)")
            except Exception as e:
                logger.error(f"Failed to load KB {name}: {e}")

        logger.info(f"CodeDescriber initialized with {len(self._plugins)} language plugins")

    def get_plugin(self, language: str) -> Optional[LanguagePlugin]:
        """Get or create a plugin for the given language."""
        language = language.lower()

        if language not in self._plugins:
            plugin = get_plugin(language)
            if plugin:
                self._plugins[language] = plugin

        return self._plugins.get(language)

    async def describe_function(
        self,
        code: str,
        language: str,
        name: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a description for a single function/routine.

        Args:
            code: The source code of the function
            language: Programming language
            name: Optional function name
            file_path: Optional source file path

        Returns:
            Generated description or None if failed
        """
        plugin = self.get_plugin(language)
        if not plugin:
            logger.error(f"No plugin for language: {language}")
            return None

        # Create a code unit
        unit = CodeUnit(
            name=name or "unknown",
            kind="function",
            language=language,
            code=code,
            file_path=file_path,
        )

        # Build prompt and generate
        prompt = plugin.build_prompt(unit)

        try:
            response = await self.orchestrator.generate_content(
                prompt=prompt,
                tier="fast"
            )

            if response and response.strip():
                return response.strip()

        except Exception as e:
            logger.error(f"Failed to generate description: {e}")

        return None

    async def describe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Extract and describe all functions in a file.

        Args:
            file_path: Path to source file
            language: Language (auto-detected if not provided)
            limit: Maximum functions to process

        Returns:
            Dict with results and statistics
        """
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        # Detect language
        if not language:
            language = detect_language(file_path)
            if not language:
                return {"error": f"Could not detect language for: {file_path}"}

        plugin = self.get_plugin(language)
        if not plugin:
            return {"error": f"No plugin for language: {language}"}

        # Read and extract
        code = path.read_text(errors='replace')
        units = plugin.extract_units(code, file_path)

        if not units:
            return {"message": "No code units found", "processed": 0}

        # Process units
        results = []
        errors = []

        for unit in units[:limit]:
            if unit.description:
                results.append({
                    "name": unit.name,
                    "description": unit.description,
                    "existing": True,
                })
                continue

            prompt = plugin.build_prompt(unit)

            try:
                response = await self.orchestrator.generate_content(
                    prompt=prompt,
                    tier="fast"
                )

                if response and response.strip():
                    unit.description = response.strip()
                    results.append({
                        "name": unit.name,
                        "kind": unit.kind,
                        "line": unit.line_number,
                        "description": unit.description,
                    })
                    logger.info(f"Described: {unit.name}")

            except Exception as e:
                errors.append(f"{unit.name}: {str(e)[:50]}")

            await asyncio.sleep(0.5)  # Rate limiting

        return {
            "file": file_path,
            "language": language,
            "total_units": len(units),
            "processed": len(results),
            "results": results,
            "errors": errors[:10],
        }

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using the orchestrator."""
        try:
            embedding = await self.orchestrator.embed_content(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    async def generate_embeddings_for_kb(
        self,
        kb_name: str,
        batch_size: int = 50,
    ) -> Dict[str, Any]:
        """Generate embeddings for all units in a knowledge base.

        Args:
            kb_name: Name of the knowledge base
            batch_size: Number of embeddings to generate per batch

        Returns:
            Statistics about embedding generation
        """
        if kb_name not in self._knowledge_bases:
            return {"error": f"Knowledge base not found: {kb_name}"}

        kb = self._knowledge_bases[kb_name]
        generated = 0
        errors = []

        for unit in kb.units:
            key = f"{unit.file_path}:{unit.name}" if unit.file_path else unit.name

            # Skip if already has embedding
            if key in kb.embeddings:
                continue

            # Build text for embedding
            text_parts = [unit.name]
            if unit.signature:
                text_parts.append(unit.signature)
            if unit.description:
                text_parts.append(unit.description)
            text_parts.append(unit.code[:300])

            text = " ".join(text_parts)

            try:
                embedding = await self.generate_embedding(text)
                if embedding:
                    kb.embeddings[key] = embedding
                    generated += 1

                    if generated % 10 == 0:
                        logger.info(f"Generated {generated} embeddings for {kb_name}")

            except Exception as e:
                errors.append(f"{unit.name}: {str(e)[:50]}")

            await asyncio.sleep(0.1)  # Rate limiting

            if generated >= batch_size:
                break

        # Save updated KB
        kb.save(self._output_dir)

        return {
            "kb_name": kb_name,
            "total_units": len(kb.units),
            "embeddings_generated": generated,
            "total_embeddings": len(kb.embeddings),
            "coverage": f"{100 * len(kb.embeddings) / max(len(kb.units), 1):.1f}%",
            "errors": errors[:10],
        }

    async def search_code(
        self,
        query: str,
        kb_name: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Search for code units by semantic similarity.

        Args:
            query: Natural language query
            kb_name: Specific KB to search (None for all)
            limit: Maximum results

        Returns:
            Search results with scores
        """
        # Generate query embedding
        query_embedding = await self.generate_embedding(query)
        if not query_embedding:
            return {"error": "Failed to generate query embedding"}

        results = []

        kbs_to_search = (
            [self._knowledge_bases[kb_name]] if kb_name and kb_name in self._knowledge_bases
            else self._knowledge_bases.values()
        )

        for kb in kbs_to_search:
            matches = kb.search(query_embedding, limit=limit)
            for unit, score in matches:
                results.append({
                    "kb": kb.name,
                    "name": unit.name,
                    "kind": unit.kind,
                    "language": unit.language,
                    "file": unit.file_path,
                    "line": unit.line_number,
                    "signature": unit.signature,
                    "description": unit.description,
                    "score": round(score, 4),
                })

        # Sort by score
        results.sort(key=lambda x: -x["score"])
        return {
            "query": query,
            "results": results[:limit],
            "total_matches": len(results),
        }

    async def describe_project(
        self,
        path: str,
        language: Optional[str] = None,
        pattern: str = "**/*",
        limit: int = 100,
        save: bool = True,
        generate_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """Describe functions across a project.

        Args:
            path: Project root directory
            language: Language filter (optional)
            pattern: Glob pattern for files
            limit: Maximum functions to process
            save: Whether to save results to disk
            generate_embeddings: Whether to generate embeddings for units

        Returns:
            Dict with results and statistics
        """
        project_path = Path(path)
        if not project_path.exists():
            return {"error": f"Path not found: {path}"}

        # Find files
        files = []
        for file_path in project_path.glob(pattern):
            if not file_path.is_file():
                continue

            detected = detect_language(str(file_path))
            if detected:
                if language is None or detected == language.lower():
                    files.append(file_path)

        if not files:
            return {"error": "No matching source files found"}

        logger.info(f"Found {len(files)} source files")

        # Create or get knowledge base for this project
        kb_name = project_path.name
        if kb_name not in self._knowledge_bases:
            self._knowledge_bases[kb_name] = CodeKnowledgeBase(name=kb_name)
        kb = self._knowledge_bases[kb_name]

        # Process files
        all_results = []
        total_processed = 0
        all_errors = []
        all_units = []

        for file_path in files:
            if total_processed >= limit:
                break

            remaining = limit - total_processed
            result = await self.describe_file(
                str(file_path),
                language=language,
                limit=remaining,
            )

            if "results" in result:
                all_results.extend(result["results"])
                total_processed += result.get("processed", 0)

                # Add units to KB
                for r in result["results"]:
                    unit = CodeUnit(
                        name=r["name"],
                        kind=r.get("kind", "function"),
                        language=language or detect_language(str(file_path)) or "unknown",
                        code="",
                        file_path=str(file_path),
                        line_number=r.get("line"),
                        description=r.get("description"),
                    )
                    all_units.append(unit)

            if "errors" in result:
                all_errors.extend(result["errors"])

        # Update KB with new units
        existing_names = {u.name for u in kb.units}
        for unit in all_units:
            if unit.name not in existing_names:
                kb.units.append(unit)

        # Save KB
        if save:
            kb.metadata["last_updated"] = str(Path(path))
            kb.metadata["total_files"] = len(files)
            kb.save(self._output_dir)
            logger.info(f"Saved KB {kb_name} with {len(kb.units)} units")

        # Generate embeddings if requested
        embedding_stats = None
        if generate_embeddings and all_units:
            embedding_stats = await self.generate_embeddings_for_kb(kb_name, batch_size=min(limit, 100))

        return {
            "project": str(project_path),
            "kb_name": kb_name,
            "files_scanned": len(files),
            "functions_described": total_processed,
            "total_kb_units": len(kb.units),
            "results": all_results,
            "errors": all_errors[:20],
            "embedding_stats": embedding_stats,
        }

    async def run_task(self, task: str = "help") -> Dict[str, Any]:
        """Run a describer task.

        Tasks:
            help - Show available tasks
            file:<path> - Describe functions in a file
            file:<path>:<lang> - Describe with explicit language
            project:<path> - Describe all functions in a project
            project:<path>:<lang> - Describe with language filter
            embed:<kb_name> - Generate embeddings for a knowledge base
            embed:<kb_name>:<batch_size> - With custom batch size
            search:<query> - Search across all knowledge bases
            search:<kb_name>:<query> - Search specific knowledge base
            kbs - List knowledge bases
            stats - Show statistics
        """
        if task == "help":
            return {
                "tasks": [
                    "file:<path> - Describe functions in a file",
                    "file:<path>:<lang> - Describe with explicit language",
                    "project:<path> - Describe all functions in project",
                    "project:<path>:<lang> - With language filter",
                    "embed:<kb_name> - Generate embeddings for KB",
                    "embed:<kb_name>:<batch_size> - With batch size",
                    "search:<query> - Search all KBs",
                    "search:<kb_name>:<query> - Search specific KB",
                    "kbs - List knowledge bases",
                    "stats - Show description statistics",
                ],
                "languages": list(set(LANGUAGE_PLUGINS.keys())),
            }

        if task.startswith("file:"):
            parts = task[5:].split(":")
            file_path = parts[0]
            language = parts[1] if len(parts) > 1 else None
            return await self.describe_file(file_path, language)

        if task.startswith("project:"):
            parts = task[8:].split(":")
            project_path = parts[0]
            language = parts[1] if len(parts) > 1 else None
            return await self.describe_project(project_path, language)

        if task.startswith("embed:"):
            parts = task[6:].split(":")
            kb_name = parts[0]
            batch_size = int(parts[1]) if len(parts) > 1 else 50
            return await self.generate_embeddings_for_kb(kb_name, batch_size)

        if task.startswith("search:"):
            parts = task[7:].split(":", 1)
            if len(parts) == 1:
                # search:<query>
                query = parts[0]
                return await self.search_code(query)
            else:
                # search:<kb_name>:<query>
                kb_name = parts[0]
                query = parts[1]
                return await self.search_code(query, kb_name)

        if task == "kbs":
            return {
                "knowledge_bases": [
                    {
                        "name": kb.name,
                        "units": len(kb.units),
                        "embeddings": len(kb.embeddings),
                        "coverage": f"{100 * len(kb.embeddings) / max(len(kb.units), 1):.1f}%",
                    }
                    for kb in self._knowledge_bases.values()
                ],
                "total_kbs": len(self._knowledge_bases),
            }

        if task == "stats":
            # Count saved descriptions
            total_units = 0
            total_embeddings = 0
            by_project = {}

            for kb in self._knowledge_bases.values():
                total_units += len(kb.units)
                total_embeddings += len(kb.embeddings)
                by_project[kb.name] = {
                    "units": len(kb.units),
                    "embeddings": len(kb.embeddings),
                }

            return {
                "total_units": total_units,
                "total_embeddings": total_embeddings,
                "by_project": by_project,
                "output_dir": str(self._output_dir),
                "plugins_available": list(self._plugins.keys()),
            }

        return {"error": f"Unknown task: {task}"}


async def main():
    """CLI entry point."""
    import sys

    describer = CodeDescriber()
    await describer.setup()

    task = sys.argv[1] if len(sys.argv) > 1 else "help"
    result = await describer.run_task(task)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
