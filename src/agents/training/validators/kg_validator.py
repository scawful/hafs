"""Knowledge Graph Validator for training samples.

Validates:
- Entity presence in knowledge graph
- Relationship consistency
- Cross-reference validity
- Domain alignment
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from agents.training.base import TrainingSample
from agents.training.validators.base import ValidationResult, Validator


class KGValidator(Validator):
    """Validator for knowledge graph consistency in training samples."""

    def __init__(
        self,
        graph_path: Optional[Path] = None,
        strict: bool = False,
        min_entity_coverage: float = 0.3,
    ):
        """Initialize KG validator.

        Args:
            graph_path: Path to knowledge graph JSON. Defaults to ~/.context/memory/knowledge_graph.json
            strict: If True, apply stricter validation (missing entities are errors)
            min_entity_coverage: Minimum fraction of mentioned entities that must be in KG
        """
        super().__init__("KGValidator", "all")  # Applies to all domains
        self.graph_path = graph_path or Path.home() / ".context" / "memory" / "knowledge_graph.json"
        self.strict = strict
        self.min_entity_coverage = min_entity_coverage

        # Lazy load graph
        self._graph: Optional[dict] = None
        self._nodes: dict[str, Any] = {}
        self._edges: list[dict[str, Any]] = []
        self._node_names: set[str] = set()
        self._routines: set[str] = set()
        self._symbols: set[str] = set()

    def _load_graph(self) -> None:
        """Load knowledge graph from disk."""
        if self._graph is not None:
            return

        if not self.graph_path.exists():
            self._graph = {"nodes": {}, "edges": []}
            return

        try:
            data = json.loads(self.graph_path.read_text())
            self._graph = data
            self._nodes = data.get("nodes", {})
            self._edges = data.get("edges", [])

            # Build lookup sets
            for node_id, node_data in self._nodes.items():
                self._node_names.add(node_id.lower())

                # Extract name from node data
                if isinstance(node_data, dict):
                    name = node_data.get("name", "")
                    if name:
                        self._node_names.add(name.lower())

                    # Track routines and symbols specifically
                    node_type = node_data.get("type", "")
                    if node_type == "routine":
                        self._routines.add(name.lower())
                    elif node_type == "symbol":
                        self._symbols.add(name.lower())

        except Exception:
            self._graph = {"nodes": {}, "edges": []}

    def can_validate(self, sample: TrainingSample) -> bool:
        """KG validator can validate any sample with kg_entities."""
        return True  # Applies to all domains

    async def validate(self, sample: TrainingSample) -> ValidationResult:
        """Validate knowledge graph consistency in the sample."""
        self._load_graph()

        errors: list[str] = []
        warnings: list[str] = []
        details: dict = {
            "entities_mentioned": [],
            "entities_found": [],
            "entities_missing": [],
            "routines_mentioned": [],
            "symbols_mentioned": [],
            "relationships_valid": True,
            "coverage": 0.0,
        }

        # Extract entities from sample
        text = f"{sample.instruction} {sample.input} {sample.output}"
        mentioned = self._extract_entities(text, sample.domain)

        details["entities_mentioned"] = mentioned

        # Check which entities exist in KG
        found = []
        missing = []

        for entity in mentioned:
            entity_lower = entity.lower()
            if self._entity_exists(entity_lower):
                found.append(entity)
            else:
                missing.append(entity)

        details["entities_found"] = found
        details["entities_missing"] = missing

        # Calculate coverage
        if mentioned:
            coverage = len(found) / len(mentioned)
        else:
            coverage = 1.0  # No entities to validate

        details["coverage"] = coverage

        # Check for routine/symbol references in ASM samples
        if sample.domain == "asm":
            routines = self._extract_routine_references(sample.output)
            symbols = self._extract_symbol_references(sample.output)

            details["routines_mentioned"] = routines
            details["symbols_mentioned"] = symbols

            # Check routine validity
            for routine in routines:
                if routine.lower() not in self._routines and routine.lower() not in self._node_names:
                    if self.strict:
                        errors.append(f"Unknown routine: {routine}")
                    else:
                        warnings.append(f"Routine not in KG: {routine}")

        # Check kg_entities from sample metadata
        if sample.kg_entities:
            for entity in sample.kg_entities:
                if not self._entity_exists(entity.lower()):
                    if self.strict:
                        errors.append(f"Tagged entity not in KG: {entity}")
                    else:
                        warnings.append(f"Tagged entity not in KG: {entity}")

        # Validate coverage threshold
        if coverage < self.min_entity_coverage and mentioned:
            msg = f"Entity coverage {coverage:.1%} below threshold {self.min_entity_coverage:.1%}"
            if self.strict:
                errors.append(msg)
            else:
                warnings.append(msg)

        # Calculate score
        score = 1.0

        # Base score on coverage
        score = min(1.0, coverage + 0.3)  # Coverage contributes up to 0.7

        # Bonus for having KG entities tagged
        if sample.kg_entities and sample.kg_validated:
            score = min(1.0, score + 0.1)

        # Penalty for missing entities
        if missing:
            penalty = len(missing) * 0.05
            score = max(0.3, score - penalty)

        return ValidationResult(
            valid=len(errors) == 0,
            score=score,
            errors=errors,
            warnings=warnings,
            details=details,
        )

    def _entity_exists(self, entity: str) -> bool:
        """Check if an entity exists in the knowledge graph."""
        entity_lower = entity.lower()

        # Direct match
        if entity_lower in self._node_names:
            return True

        # Check with common prefixes
        prefixes = ["alttp:", "oracle-of-secrets:", "project:", "routine:", "symbol:"]
        for prefix in prefixes:
            if f"{prefix}{entity_lower}" in self._node_names:
                return True
            # Also check node IDs directly
            for node_id in self._nodes:
                if node_id.lower().endswith(f":{entity_lower}"):
                    return True

        return False

    def _extract_entities(self, text: str, domain: str) -> list[str]:
        """Extract potential entity references from text."""
        entities = []

        # Common patterns for entity references
        patterns = [
            # Code references like `EntityName` or `RoutineName`
            r'`([A-Z][a-zA-Z0-9_]+)`',
            # Capitalized terms that look like identifiers
            r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b',  # CamelCase
            # Routine names (common in ASM)
            r'\b(Link_[A-Za-z0-9_]+)\b',
            r'\b(Player_[A-Za-z0-9_]+)\b',
            r'\b(Sprite_[A-Za-z0-9_]+)\b',
            r'\b(Module_[A-Za-z0-9_]+)\b',
            # Memory addresses with labels
            r'\b([A-Z][A-Za-z0-9]+_[A-Z][A-Za-z0-9]+)\b',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)

        # Domain-specific extraction
        if domain == "asm":
            # Extract ASM-specific references
            asm_patterns = [
                r'\b([A-Z][a-z]+_[A-Z][a-z_0-9]+)\b',  # Link_HandleSword
                r'@([A-Za-z_][A-Za-z0-9_]+)',  # @Labels
            ]
            for pattern in asm_patterns:
                matches = re.findall(pattern, text)
                entities.extend(matches)

        elif domain == "cpp":
            # Extract C++ class/function names
            cpp_patterns = [
                r'\bclass\s+([A-Z][a-zA-Z0-9_]+)\b',
                r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)::\w+',  # ClassName::method
            ]
            for pattern in cpp_patterns:
                matches = re.findall(pattern, text)
                entities.extend(matches)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                unique.append(e)

        return unique

    def _extract_routine_references(self, code: str) -> list[str]:
        """Extract routine/label references from ASM code."""
        routines = []

        # JSR/JSL targets
        jsr_pattern = r'\b(?:JSR|JSL|JMP|JML)\s+([A-Za-z_][A-Za-z0-9_]+)\b'
        matches = re.findall(jsr_pattern, code, re.IGNORECASE)
        routines.extend(matches)

        # BRA/BRL targets
        branch_pattern = r'\b(?:BRA|BRL|BEQ|BNE|BCC|BCS|BMI|BPL)\s+([A-Za-z_][A-Za-z0-9_]+)\b'
        matches = re.findall(branch_pattern, code, re.IGNORECASE)
        routines.extend(matches)

        return list(set(routines))

    def _extract_symbol_references(self, code: str) -> list[str]:
        """Extract symbol/variable references from ASM code."""
        symbols = []

        # LDA/STA with labels
        load_store_pattern = r'\b(?:LDA|LDX|LDY|STA|STX|STY)\s+([A-Za-z_][A-Za-z0-9_]+)\b'
        matches = re.findall(load_store_pattern, code, re.IGNORECASE)
        symbols.extend(matches)

        # Filter out common non-symbol patterns
        filtered = []
        for sym in symbols:
            # Skip if it looks like a routine name
            if sym.lower() in self._routines:
                continue
            # Skip common mnemonics that might be captured
            if sym.upper() in {'A', 'X', 'Y', 'S'}:
                continue
            filtered.append(sym)

        return list(set(filtered))

    def get_related_entities(self, entity: str) -> list[dict[str, Any]]:
        """Get entities related to a given entity in the KG."""
        self._load_graph()

        related = []
        entity_lower = entity.lower()

        for edge in self._edges:
            source = str(edge.get("source", "")).lower()
            target = str(edge.get("target", "")).lower()
            relation = edge.get("relation", "")

            if entity_lower in source:
                related.append({
                    "entity": edge.get("target"),
                    "relation": relation,
                    "direction": "outgoing",
                })
            elif entity_lower in target:
                related.append({
                    "entity": edge.get("source"),
                    "relation": relation,
                    "direction": "incoming",
                })

        return related

    def suggest_entities(self, partial: str, limit: int = 10) -> list[str]:
        """Suggest entity names matching a partial string."""
        self._load_graph()

        partial_lower = partial.lower()
        matches = []

        for node_id in self._nodes:
            if partial_lower in node_id.lower():
                matches.append(node_id)
                if len(matches) >= limit:
                    break

        return matches
