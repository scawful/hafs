# Code Describer Architecture

## Overview

The CodeDescriber system provides multi-language code analysis with LLM-powered description generation and semantic search via embeddings.

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CodeDescriber                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │ Language Plugins│  │ Knowledge Bases │  │ LLM Integration│  │
│  │ (regex-based)   │  │ (JSON + embeds) │  │ (Gemini API)   │  │
│  └────────┬────────┘  └────────┬────────┘  └───────┬────────┘  │
│           │                    │                    │           │
│           ▼                    ▼                    ▼           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Processing Pipeline                      ││
│  │  1. Extract code units (via plugin)                         ││
│  │  2. Generate descriptions (via LLM)                         ││
│  │  3. Generate embeddings (via Gemini)                        ││
│  │  4. Store in KB (JSON files)                                ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Supported Languages (11)

| Language | Extensions | Key Features |
|----------|-----------|--------------|
| Assembly | .asm, .s, .inc | Label/routine detection, RTS/RTL boundaries |
| C/C++ | .c, .cpp, .h, .hpp | Functions, methods, class detection |
| Python | .py | Functions with docstring extraction |
| Java | .java | Methods, class context |
| Kotlin | .kt, .kts | Extension functions, suspend detection |
| Swift | .swift | Functions, initializers, async/throws |
| Rust | .rs | Functions with ownership hints |
| Go | .go | Functions, methods with receivers |
| TypeScript/JS | .ts, .tsx, .js | Functions, arrow functions |
| Lisp | .lisp, .scm | defun/define patterns |

---

## Identified Improvements

### 1. AST-Based Parsing (High Priority)

**Problem**: Regex-based extraction is fragile and misses edge cases.

**Solution**: Use language-specific AST parsers:

```python
class ASTPlugin(LanguagePlugin):
    """Base class for AST-based plugins."""

    @abstractmethod
    def get_parser(self) -> Any:
        """Return the AST parser for this language."""
        pass

    def extract_units(self, code: str, file_path: Optional[str] = None) -> List[CodeUnit]:
        tree = self.get_parser().parse(code)
        return self._walk_ast(tree, file_path)

# Suggested parsers:
# - Python: ast module (stdlib)
# - TypeScript/JS: tree-sitter-javascript
# - C/C++: tree-sitter-cpp or libclang
# - Rust: tree-sitter-rust or syn
# - Swift: SwiftSyntax (via subprocess)
# - Go: go/parser (via subprocess)
```

### 2. Incremental Processing (High Priority)

**Problem**: Re-processing entire projects is slow and wasteful.

**Solution**: Track file hashes and only process changed files:

```python
@dataclass
class CodeKnowledgeBase:
    file_hashes: Dict[str, str] = field(default_factory=dict)  # path -> hash

    def needs_update(self, file_path: str) -> bool:
        current_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()
        return self.file_hashes.get(file_path) != current_hash

    def mark_processed(self, file_path: str):
        self.file_hashes[file_path] = hashlib.md5(
            Path(file_path).read_bytes()
        ).hexdigest()
```

### 3. Parallel Processing (Medium Priority)

**Problem**: Sequential LLM calls are slow for large projects.

**Solution**: Use asyncio.gather with rate limiting:

```python
async def describe_batch(self, units: List[CodeUnit], max_concurrent: int = 5):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def describe_with_limit(unit):
        async with semaphore:
            return await self.describe_unit(unit)

    tasks = [describe_with_limit(u) for u in units]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

### 4. Cross-Reference Detection (Medium Priority)

**Problem**: We don't track who calls what.

**Solution**: Build call graph during extraction:

```python
@dataclass
class CodeUnit:
    calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)

def build_call_graph(self, units: List[CodeUnit]):
    """Analyze code to build call relationships."""
    name_to_unit = {u.name: u for u in units}

    for unit in units:
        # Find function calls in code
        for called_name in self._find_calls(unit.code):
            if called_name in name_to_unit:
                unit.calls.append(called_name)
                name_to_unit[called_name].called_by.append(unit.name)
```

### 5. Documentation Extraction (Medium Priority)

**Problem**: Existing docstrings/comments are ignored.

**Solution**: Extract and include in embeddings:

```python
class PythonPlugin(LanguagePlugin):
    def extract_docstring(self, code: str) -> Optional[str]:
        match = re.search(r'"""(.+?)"""', code, re.DOTALL)
        return match.group(1).strip() if match else None

class CppPlugin(LanguagePlugin):
    def extract_doxygen(self, code: str, func_pos: int) -> Optional[str]:
        # Look backwards for /** ... */ or /// comments
        before = code[:func_pos]
        match = re.search(r'/\*\*(.+?)\*/', before[-500:], re.DOTALL)
        return match.group(1).strip() if match else None
```

### 6. LSP Integration (Low Priority, High Impact)

**Problem**: Manual parsing is duplicating work LSP servers already do.

**Solution**: Query language servers for symbols:

```python
class LSPPlugin(LanguagePlugin):
    """Plugin that uses Language Server Protocol for extraction."""

    async def extract_units_via_lsp(self, file_path: str) -> List[CodeUnit]:
        # Send textDocument/documentSymbol request
        symbols = await self.lsp_client.document_symbols(file_path)

        units = []
        for symbol in symbols:
            if symbol.kind in [SymbolKind.Function, SymbolKind.Method]:
                units.append(CodeUnit(
                    name=symbol.name,
                    kind=symbol.kind.name.lower(),
                    line_number=symbol.range.start.line,
                    # LSP provides accurate ranges
                ))
        return units
```

### 7. Configurable Prompts (Low Priority)

**Problem**: Prompts are hardcoded in plugins.

**Solution**: Allow per-project prompt customization:

```python
# ~/.context/code_descriptions/yaze/config.yaml
prompts:
  cpp:
    function: |
      Analyze this yaze emulator function. Focus on:
      - SNES hardware interaction
      - Memory management
      - Performance considerations

      Function: {name}
      Code: {code}
```

### 8. Embedding Clustering (Enhancement)

**Problem**: Can't visualize code organization.

**Solution**: Add clustering for KB exploration:

```python
async def cluster_units(self, kb_name: str, n_clusters: int = 10):
    """Cluster code units by embedding similarity."""
    kb = self._knowledge_bases[kb_name]

    embeddings = np.array([kb.embeddings[k] for k in kb.embeddings])

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(embeddings)

    clusters = defaultdict(list)
    for i, (key, _) in enumerate(kb.embeddings.items()):
        clusters[labels[i]].append(key)

    return dict(clusters)
```

---

## Plugin Development Guide

### Creating a New Plugin

```python
from hafs.agents.code_describer import LanguagePlugin, CodeUnit, register_plugin

class MyLanguagePlugin(LanguagePlugin):
    name = "mylang"
    extensions = [".ml", ".mli"]

    # Define patterns for your language
    FUNCTION_PATTERN = re.compile(r'let\s+(\w+)\s*=')

    def extract_units(self, code: str, file_path: Optional[str] = None) -> List[CodeUnit]:
        units = []
        for match in self.FUNCTION_PATTERN.finditer(code):
            units.append(CodeUnit(
                name=match.group(1),
                kind="function",
                language=self.name,
                code=code[match.start():match.start()+500],
                file_path=file_path,
                line_number=code[:match.start()].count('\n') + 1,
            ))
        return units

    def build_prompt(self, unit: CodeUnit) -> str:
        return f"""Analyze this {self.name} function: {unit.name}

Code:
{unit.code[:800]}

Respond with a 1-2 sentence description."""

    def get_context_hints(self) -> List[str]:
        return ["Look for pattern matching", "Note type annotations"]

# Register the plugin
register_plugin("mylang", MyLanguagePlugin)
register_plugin("ml", MyLanguagePlugin)  # Alias
```

---

## Future Roadmap

1. **v1.1**: Incremental processing + parallel LLM calls
2. **v1.2**: AST-based parsing for Python/TypeScript
3. **v1.3**: Cross-reference detection and call graphs
4. **v1.4**: LSP integration for accurate symbol extraction
5. **v2.0**: Full code intelligence with semantic understanding
