"""Prompt template rotation system for diverse training sample generation.

Provides templates that vary perspective, tone, complexity, and context to
increase embedding diversity and reduce rejection rates.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import toml


@dataclass
class PromptTemplate:
    """A single prompt template with metadata."""

    template: str  # Template string with {placeholders}
    category: str  # perspective/tone/complexity/context
    domain: str  # asm/oracle/cpp/text
    tags: list[str] = field(default_factory=list)  # Additional metadata

    def render(self, **kwargs: Any) -> str:
        """Render template with provided values."""
        return self.template.format(**kwargs)


class PromptTemplateRotator:
    """Rotates through diverse prompt templates to ensure variety.

    Tracks usage counts and returns least-used templates to maximize
    diversity in generated samples.
    """

    def __init__(self, domain: str, config_path: Optional[Path] = None):
        """Initialize template rotator for a domain.

        Args:
            domain: Domain name (asm, oracle, cpp, etc.)
            config_path: Path to prompt_templates.toml (optional)
        """
        self.domain = domain
        self.config_path = config_path or self._get_default_config_path()
        self.templates: list[PromptTemplate] = []
        self.usage_counts: dict[str, int] = {}

        self._load_templates()

    def _get_default_config_path(self) -> Path:
        """Get default config path."""
        return Path(__file__).parent.parent.parent.parent / "config" / "prompt_templates.toml"

    def _load_templates(self):
        """Load templates from TOML config."""
        if not self.config_path.exists():
            # Use built-in defaults if config doesn't exist
            self._load_builtin_templates()
            return

        config = toml.load(self.config_path)

        if self.domain not in config:
            # Fall back to built-in if domain not in config
            self._load_builtin_templates()
            return

        domain_config = config[self.domain]

        for category, templates in domain_config.items():
            if not isinstance(templates, list):
                continue

            for template_str in templates:
                template = PromptTemplate(
                    template=template_str,
                    category=category,
                    domain=self.domain,
                )
                self.templates.append(template)
                # Initialize usage count
                self.usage_counts[template.template] = 0

    def _load_builtin_templates(self):
        """Load built-in default templates."""
        if self.domain == "asm":
            self._load_asm_templates()
        elif self.domain == "oracle":
            self._load_oracle_templates()
        elif self.domain == "cpp":
            self._load_cpp_templates()
        else:
            # Generic templates
            self._load_generic_templates()

    def _load_asm_templates(self):
        """Load built-in ASM templates."""
        asm_templates = [
            # Perspective variations
            ("Write assembly code to {action}", "perspective"),
            ("Implement {feature} in 65816 assembly", "perspective"),
            ("Show how to {action} using SNES hardware", "perspective"),

            # Tone variations
            ("Optimize this routine to {goal}", "tone"),
            ("Refactor {routine} for better {quality}", "tone"),
            ("Debug this code that {issue}", "tone"),

            # Complexity variations
            ("Explain step-by-step how to {task}", "complexity"),  # Beginner
            ("Write production-quality code for {feature}", "complexity"),  # Expert
            ("Provide concise reference code for {feature}", "complexity"),  # Reference

            # Context variations
            ("Implement {feature} for dungeon context using {registers}", "context"),
            ("Write {routine} that interacts with PPU/DMA", "context"),
            ("Create NMI-safe code for {feature}", "context"),

            # Constraint variations
            ("Implement {feature} using only {n} bytes of RAM", "constraint"),
            ("Write {routine} that fits in bank ${bank}", "constraint"),
            ("Optimize {routine} for minimal CPU cycles", "constraint"),

            # Comparison variations
            ("Compare different approaches for {problem}", "comparison"),
            ("Show vanilla ALTTP code vs optimized version for {feature}", "comparison"),
            ("Explain trade-offs between {approach1} and {approach2}", "comparison"),
        ]

        for template_str, category in asm_templates:
            template = PromptTemplate(
                template=template_str,
                category=category,
                domain="asm",
            )
            self.templates.append(template)
            self.usage_counts[template.template] = 0

    def _load_oracle_templates(self):
        """Load built-in Oracle ROM hack templates."""
        oracle_templates = [
            # Hook explanations
            ("Explain how Oracle hooks {vanilla_routine}", "hook"),
            ("Show the JSL hook for {feature} in Oracle", "hook"),
            ("Compare vanilla vs Oracle implementation of {system}", "hook"),

            # ROM hacking techniques
            ("How to add {feature} to your ROM hack", "technique"),
            ("Explain the ROM hacking technique for {feature}", "technique"),
            ("Show bank allocation strategy for {feature}", "technique"),

            # Integration
            ("Why does Oracle use bank ${bank} for {feature}?", "integration"),
            ("How does {system} integrate with {other_system}?", "integration"),
            ("Explain the call graph for {feature}", "integration"),

            # Testing
            ("How to test {modification} in-game", "testing"),
            ("What edge cases should be tested for {feature}?", "testing"),
            ("Verify {feature} works correctly with vanilla mechanics", "testing"),

            # Alternatives
            ("What other ways could {feature} be implemented?", "alternatives"),
            ("Compare pushpc/pullpc vs org for {modification}", "alternatives"),
            ("Discuss trade-offs of {approach} for {feature}", "alternatives"),
        ]

        for template_str, category in oracle_templates:
            template = PromptTemplate(
                template=template_str,
                category=category,
                domain="oracle",
            )
            self.templates.append(template)
            self.usage_counts[template.template] = 0

    def _load_cpp_templates(self):
        """Load built-in C++ YAZE templates."""
        cpp_templates = [
            # API usage
            ("Use YAZE API to {action}", "api"),
            ("Implement {feature} using {yaze_class}", "api"),
            ("Show YAZE workflow for {task}", "api"),

            # Code quality
            ("Write production-quality C++ for {feature}", "quality"),
            ("Refactor this code with proper const correctness for {feature}", "quality"),
            ("Add error handling to {feature}", "quality"),

            # Algorithms
            ("Explain the algorithm for {operation} in YAZE", "algorithm"),
            ("Optimize {routine} for time complexity", "algorithm"),
            ("Show efficient implementation of {data_structure}", "algorithm"),

            # Integration
            ("How does {class} integrate with ROM editor?", "integration"),
            ("Connect {feature} to graphics pipeline", "integration"),
            ("Implement {feature} that works with existing {system}", "integration"),
        ]

        for template_str, category in cpp_templates:
            template = PromptTemplate(
                template=template_str,
                category=category,
                domain="cpp",
            )
            self.templates.append(template)
            self.usage_counts[template.template] = 0

    def _load_generic_templates(self):
        """Load generic templates for unknown domains."""
        generic_templates = [
            ("Explain how to {action}", "generic"),
            ("Implement {feature}", "generic"),
            ("Show code for {task}", "generic"),
            ("Optimize {feature} for {goal}", "generic"),
        ]

        for template_str, category in generic_templates:
            template = PromptTemplate(
                template=template_str,
                category=category,
                domain=self.domain,
            )
            self.templates.append(template)
            self.usage_counts[template.template] = 0

    def get_next_template(self, category: Optional[str] = None) -> PromptTemplate:
        """Get least-used template (balanced rotation).

        Args:
            category: Optional category filter (perspective, tone, etc.)

        Returns:
            PromptTemplate with lowest usage count
        """
        if not self.templates:
            raise ValueError(f"No templates loaded for domain: {self.domain}")

        # Filter by category if specified
        candidates = self.templates
        if category:
            candidates = [t for t in self.templates if t.category == category]
            if not candidates:
                # Fall back to all templates if category filter yields nothing
                candidates = self.templates

        # Find minimum usage count
        min_usage = min(self.usage_counts[t.template] for t in candidates)

        # Get all templates with minimum usage
        least_used = [t for t in candidates if self.usage_counts[t.template] == min_usage]

        # Random choice among least-used
        template = random.choice(least_used)

        # Increment usage count
        self.usage_counts[template.template] += 1

        return template

    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dict with template counts, usage distribution, etc.
        """
        return {
            "domain": self.domain,
            "total_templates": len(self.templates),
            "total_uses": sum(self.usage_counts.values()),
            "usage_counts": dict(self.usage_counts),
            "categories": list(set(t.category for t in self.templates)),
        }

    def reset_counts(self):
        """Reset all usage counts to zero."""
        for key in self.usage_counts:
            self.usage_counts[key] = 0
