"""C++ Validator for training samples.

Validates:
- Basic syntax checks (brackets, braces, semicolons)
- Keyword usage
- Common patterns
- Optional: Compile check with clang (if available)
"""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from agents.training.base import TrainingSample
from agents.training.validators.base import ValidationResult, Validator


class CppValidator(Validator):
    """Validator for C++ code in training samples."""

    # C++ keywords
    KEYWORDS = {
        # Storage class
        "auto", "register", "static", "extern", "mutable", "thread_local",
        # Type specifiers
        "void", "bool", "char", "short", "int", "long", "float", "double",
        "signed", "unsigned", "wchar_t", "char8_t", "char16_t", "char32_t",
        # Type qualifiers
        "const", "volatile", "constexpr", "consteval", "constinit",
        # Control flow
        "if", "else", "switch", "case", "default", "while", "do", "for",
        "break", "continue", "return", "goto",
        # Declarations
        "class", "struct", "union", "enum", "typedef", "using", "namespace",
        "template", "typename", "concept", "requires",
        # Access specifiers
        "public", "private", "protected",
        # Other keywords
        "virtual", "override", "final", "explicit", "inline", "friend",
        "operator", "sizeof", "alignof", "decltype", "typeid",
        "new", "delete", "this", "nullptr", "true", "false",
        "try", "catch", "throw", "noexcept",
        "static_assert", "static_cast", "dynamic_cast", "const_cast", "reinterpret_cast",
        "co_await", "co_return", "co_yield",
        # Modules (C++20)
        "module", "import", "export",
    }

    # Common C++ standard library types
    STD_TYPES = {
        "string", "vector", "map", "unordered_map", "set", "unordered_set",
        "list", "deque", "array", "pair", "tuple", "optional", "variant",
        "shared_ptr", "unique_ptr", "weak_ptr", "function", "any",
        "thread", "mutex", "lock_guard", "unique_lock", "condition_variable",
        "future", "promise", "async", "atomic",
        "ifstream", "ofstream", "fstream", "stringstream", "ostringstream",
        "iostream", "cin", "cout", "cerr", "endl",
        "size_t", "ptrdiff_t", "nullptr_t", "byte",
        "int8_t", "int16_t", "int32_t", "int64_t",
        "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    }

    def __init__(
        self,
        check_compile: bool = False,
        compiler: str = "clang++",
        strict: bool = False,
    ):
        """Initialize C++ validator.

        Args:
            check_compile: If True, attempt to compile the code
            compiler: Compiler to use for compile checks
            strict: If True, apply stricter validation
        """
        super().__init__("CppValidator", "cpp")
        self.check_compile = check_compile
        self.compiler = compiler
        self.strict = strict

        # Check if compiler is available
        self._compiler_available = shutil.which(compiler) is not None

    async def validate(self, sample: TrainingSample) -> ValidationResult:
        """Validate C++ code in the sample output."""
        errors: list[str] = []
        warnings: list[str] = []
        details: dict = {
            "syntax_issues": [],
            "keywords_found": [],
            "std_types_found": [],
            "bracket_balance": True,
            "compile_checked": False,
            "compile_result": None,
        }

        code = sample.output

        # Basic syntax checks
        syntax_result = self._check_syntax(code)
        details["syntax_issues"] = syntax_result["issues"]
        details["bracket_balance"] = syntax_result["balanced"]

        if not syntax_result["balanced"]:
            errors.append("Unbalanced brackets/braces/parentheses")

        for issue in syntax_result["issues"]:
            if self.strict:
                errors.append(issue)
            else:
                warnings.append(issue)

        # Check for keywords and types
        details["keywords_found"] = self._find_keywords(code)
        details["std_types_found"] = self._find_std_types(code)

        # Compile check if enabled and available
        if self.check_compile and self._compiler_available:
            compile_result = await self._check_compile(code)
            details["compile_checked"] = True
            details["compile_result"] = compile_result

            if not compile_result["success"]:
                if self.strict:
                    errors.append(f"Compile error: {compile_result['error'][:200]}")
                else:
                    warnings.append(f"Compile warning: {compile_result['error'][:100]}")

        # Calculate score
        score = 1.0

        # Deduct for syntax issues
        score -= len(details["syntax_issues"]) * 0.1
        score = max(0.0, score)

        # Deduct for bracket imbalance
        if not details["bracket_balance"]:
            score -= 0.3

        # Bonus for using C++ features
        if details["keywords_found"]:
            score = min(1.0, score + 0.05)
        if details["std_types_found"]:
            score = min(1.0, score + 0.05)

        # Deduct for compile failure
        if details["compile_checked"] and not details["compile_result"]["success"]:
            score -= 0.2

        score = max(0.0, min(1.0, score))

        return ValidationResult(
            valid=len(errors) == 0,
            score=score,
            errors=errors,
            warnings=warnings,
            details=details,
        )

    def _check_syntax(self, code: str) -> dict:
        """Check basic C++ syntax."""
        issues = []
        balanced = True

        # Check bracket balance
        stack = []
        pairs = {"(": ")", "[": "]", "{": "}"}
        in_string = False
        in_char = False
        in_comment = False
        in_block_comment = False

        i = 0
        while i < len(code):
            c = code[i]

            # Handle comments
            if not in_string and not in_char:
                if i < len(code) - 1:
                    two_char = code[i:i+2]
                    if two_char == "//":
                        # Skip to end of line
                        while i < len(code) and code[i] != "\n":
                            i += 1
                        continue
                    elif two_char == "/*":
                        in_block_comment = True
                        i += 2
                        continue
                    elif two_char == "*/" and in_block_comment:
                        in_block_comment = False
                        i += 2
                        continue

            if in_block_comment:
                i += 1
                continue

            # Handle strings
            if c == '"' and not in_char and (i == 0 or code[i-1] != '\\'):
                in_string = not in_string
            elif c == "'" and not in_string and (i == 0 or code[i-1] != '\\'):
                in_char = not in_char

            if not in_string and not in_char:
                if c in pairs:
                    stack.append(c)
                elif c in pairs.values():
                    if not stack:
                        balanced = False
                        issues.append(f"Unexpected closing bracket '{c}'")
                    else:
                        expected = pairs[stack.pop()]
                        if c != expected:
                            balanced = False
                            issues.append(f"Mismatched brackets: expected '{expected}', got '{c}'")

            i += 1

        if stack:
            balanced = False
            issues.append(f"Unclosed brackets: {stack}")

        # Check for common issues
        # Missing semicolons after statements (heuristic)
        lines = code.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip empty lines, comments, preprocessor
            if not stripped or stripped.startswith("//") or stripped.startswith("#"):
                continue

            # Skip lines that end with block characters
            if stripped.endswith("{") or stripped.endswith("}") or stripped.endswith(":"):
                continue

            # Skip lines that are likely continuations
            if stripped.endswith(",") or stripped.endswith("\\"):
                continue

            # Check for statements that should end with semicolon
            # This is a heuristic and may have false positives
            statement_patterns = [
                r"return\s+.+[^;]$",  # return without semicolon
                r"break$",  # break without semicolon
                r"continue$",  # continue without semicolon
            ]

            for pattern in statement_patterns:
                if re.search(pattern, stripped):
                    issues.append(f"Line {i+1}: Possibly missing semicolon")
                    break

        return {"issues": issues, "balanced": balanced}

    def _find_keywords(self, code: str) -> list[str]:
        """Find C++ keywords in code."""
        found = []
        # Use word boundaries to find keywords
        for keyword in self.KEYWORDS:
            if re.search(rf"\b{keyword}\b", code):
                found.append(keyword)
        return found

    def _find_std_types(self, code: str) -> list[str]:
        """Find standard library types in code."""
        found = []
        for type_name in self.STD_TYPES:
            # Check for std::type or just type in common contexts
            if re.search(rf"std::{type_name}\b", code) or re.search(rf"\b{type_name}<", code):
                found.append(type_name)
        return found

    async def _check_compile(self, code: str) -> dict:
        """Attempt to compile the code."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cpp", delete=False
        ) as f:
            # Add minimal includes for standalone compilation
            wrapped_code = """
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

// Sample code below
""" + code
            f.write(wrapped_code)
            temp_path = Path(f.name)

        try:
            # Run compiler with syntax-only check
            process = await asyncio.create_subprocess_exec(
                self.compiler,
                "-fsyntax-only",
                "-std=c++17",
                "-Wall",
                str(temp_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=10.0
            )

            success = process.returncode == 0
            error = stderr.decode("utf-8", errors="replace") if stderr else ""

            return {
                "success": success,
                "error": error,
                "returncode": process.returncode,
            }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Compilation timed out",
                "returncode": -1,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "returncode": -1,
            }
        finally:
            # Clean up temp file
            try:
                temp_path.unlink()
            except Exception:
                pass
