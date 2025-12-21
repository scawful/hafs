"""Quality Pipeline for training data refinement.

Provides scoring, deduplication, and validation for training samples
using embeddings, knowledge graph, hallucination detection, and
domain-specific validators with feedback tracking.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from agents.training.base import QualityScore, TrainingSample

logger = logging.getLogger(__name__)


# Lazy imports for validators and feedback
def _get_validators():
    """Lazy import validators to avoid circular imports."""
    try:
        from agents.training.validators import (
            AsmValidator,
            CppValidator,
            KGValidator,
            CompositeValidator,
        )
        return AsmValidator, CppValidator, KGValidator, CompositeValidator
    except ImportError:
        return None, None, None, None


def _get_feedback_tracker():
    """Lazy import feedback tracker."""
    try:
        from agents.training.feedback import QualityFeedbackTracker, RejectionReason
        return QualityFeedbackTracker, RejectionReason
    except ImportError:
        return None, None


def _get_active_learner():
    """Lazy import active learning sampler."""
    try:
        from agents.training.active_learning import ActiveLearningSampler
        return ActiveLearningSampler
    except ImportError:
        return None


@dataclass
class DuplicateResult:
    """Result of duplicate detection."""

    is_duplicate: bool
    similarity: float
    matched_id: Optional[str] = None


@dataclass
class FilterStats:
    """Statistics from a filter_samples run."""

    total: int
    accepted: int
    rejected_validation: int
    rejected_quality: int
    rejected_duplicates: int

    @property
    def passed_quality(self) -> int:
        return self.total - self.rejected_validation - self.rejected_quality


class QualityPipeline:
    """Quality scoring and refinement for training samples.

    Integrates with:
    - StreamingIndex for embedding-based deduplication
    - KnowledgeGraphAgent for entity validation
    - UnifiedOrchestrator for hallucination detection
    - Domain-specific validators (ASM, C++, KG)
    - Quality feedback tracking
    - Active learning for coverage optimization
    """

    # Thresholds
    SIMILARITY_THRESHOLD = 0.95  # Max similarity for dedup
    MIN_QUALITY_SCORE = 0.7  # Minimum acceptable quality

    def __init__(
        self,
        embedding_index: Optional[Any] = None,
        kg_agent: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
        enable_validators: bool = True,
        enable_feedback: bool = True,
        enable_active_learning: bool = True,
    ):
        """Initialize quality pipeline.

        Args:
            embedding_index: StreamingIndex for deduplication
            kg_agent: KnowledgeGraphAgent for validation
            orchestrator: UnifiedOrchestrator for hallucination detection
            enable_validators: Whether to use domain-specific validators
            enable_feedback: Whether to track quality feedback
            enable_active_learning: Whether to use active learning sampler
        """
        self.embedding_index = embedding_index
        self.kg_agent = kg_agent
        self.orchestrator = orchestrator
        self._kg_graph: Optional[dict[str, Any]] = None

        # Domain validators
        self._validators: dict[str, Any] = {}
        self._composite_validator: Optional[Any] = None
        self._enable_validators = enable_validators

        # Feedback tracking
        self._feedback_tracker: Optional[Any] = None
        self._enable_feedback = enable_feedback
        self._RejectionReason: Optional[type] = None

        # Active learning
        self._active_learner: Optional[Any] = None
        self._enable_active_learning = enable_active_learning

        # Last filter run stats
        self.last_filter_stats: Optional[FilterStats] = None

    async def setup(self):
        """Initialize components if not provided."""
        if self.embedding_index is None:
            try:
                from hafs.core.streaming_index import StreamingIndex

                self.embedding_index = StreamingIndex(dim=768, max_elements=100000)
            except ImportError:
                logger.warning("StreamingIndex not available")

        if self.kg_agent is None:
            try:
                from agents.knowledge.graph import KnowledgeGraphAgent

                self.kg_agent = KnowledgeGraphAgent()
            except ImportError:
                logger.warning("KnowledgeGraphAgent not available")

        if self.orchestrator is None:
            try:
                from hafs.core.orchestrator_v2 import UnifiedOrchestrator

                self.orchestrator = UnifiedOrchestrator()
            except ImportError:
                logger.warning("UnifiedOrchestrator not available")

        # Initialize validators
        if self._enable_validators:
            self._setup_validators()

        # Initialize feedback tracker
        if self._enable_feedback:
            self._setup_feedback_tracker()

        # Initialize active learning
        if self._enable_active_learning:
            self._setup_active_learner()

    def _setup_validators(self) -> None:
        """Initialize domain-specific validators."""
        AsmValidator, CppValidator, KGValidator, CompositeValidator = _get_validators()

        if AsmValidator is None:
            logger.warning("Validators not available")
            return

        # Create domain validators
        self._validators = {
            "asm": AsmValidator(strict=False),
            "cpp": CppValidator(check_compile=False, strict=False),
        }

        # KG validator applies to all domains
        kg_validator = KGValidator(strict=False)

        # Create composite validator
        all_validators = list(self._validators.values()) + [kg_validator]
        self._composite_validator = CompositeValidator(all_validators)

        logger.info(f"Initialized {len(self._validators)} domain validators")

    def _setup_feedback_tracker(self) -> None:
        """Initialize feedback tracker."""
        QualityFeedbackTracker, RejectionReason = _get_feedback_tracker()

        if QualityFeedbackTracker is None:
            logger.warning("Feedback tracker not available")
            return

        self._feedback_tracker = QualityFeedbackTracker(auto_adjust_thresholds=True)
        self._RejectionReason = RejectionReason

        # Apply any learned thresholds
        if self._feedback_tracker.thresholds:
            self.MIN_QUALITY_SCORE = self._feedback_tracker.thresholds.get(
                "min_quality_score", self.MIN_QUALITY_SCORE
            )
            self.SIMILARITY_THRESHOLD = self._feedback_tracker.thresholds.get(
                "similarity_threshold", self.SIMILARITY_THRESHOLD
            )

        logger.info("Initialized feedback tracker")

    def _setup_active_learner(self) -> None:
        """Initialize active learning sampler."""
        ActiveLearningSampler = _get_active_learner()

        if ActiveLearningSampler is None:
            logger.warning("Active learning sampler not available")
            return

        self._active_learner = ActiveLearningSampler(embedding_dim=768)
        logger.info("Initialized active learning sampler")

    async def validate(self, sample: TrainingSample) -> tuple[bool, dict[str, Any]]:
        """Run domain-specific validation on a sample.

        Args:
            sample: Training sample to validate

        Returns:
            Tuple of (is_valid, validation_details)
        """
        if not self._composite_validator:
            return True, {}

        try:
            result = await self._composite_validator.validate(sample)
            return result.valid, result.to_dict()
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return True, {"error": str(e)}

    async def score(self, sample: TrainingSample) -> QualityScore:
        """Compute quality score for a sample.

        Args:
            sample: Training sample to score

        Returns:
            QualityScore with component scores
        """
        diversity = await self._score_diversity(sample)
        kg_consistency = await self._validate_kg(sample)
        hallucination = await self._check_hallucination(sample)
        coherence = self._score_coherence(sample)

        return QualityScore(
            diversity_score=diversity,
            kg_consistency=kg_consistency,
            hallucination_risk=hallucination,
            semantic_coherence=coherence,
        )

    async def _score_diversity(self, sample: TrainingSample) -> float:
        """Score diversity based on embedding distance from existing samples."""
        if self.embedding_index is None or self.embedding_index.size() == 0:
            return 1.0  # Max diversity if no comparisons possible

        # Get or generate embedding
        if sample.embedding is None:
            if self.orchestrator:
                try:
                    sample.embedding = await self.orchestrator.embed(sample.instruction)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")
                    return 1.0
            else:
                return 1.0

        # Search for similar samples
        try:
            query = np.array(sample.embedding, dtype=np.float32)
            ids, scores = self.embedding_index.search(query, k=1)

            if len(scores) > 0:
                # Convert similarity to diversity (1 - similarity)
                similarity = float(scores[0])
                return 1.0 - similarity
            return 1.0
        except Exception as e:
            logger.warning(f"Diversity scoring failed: {e}")
            return 1.0

    async def _validate_kg(self, sample: TrainingSample) -> float:
        """Check sample consistency with knowledge graph.

        Skips SNES hardware registers (they're not in the KG).
        """
        if self.kg_agent is None:
            return 1.0  # Assume valid if no KG available

        # Load graph if not cached
        if self._kg_graph is None:
            try:
                self._kg_graph = await self.kg_agent.build_graph()
            except Exception as e:
                logger.warning(f"Failed to load knowledge graph: {e}")
                return 1.0

        # SNES hardware registers (should NOT be checked against KG)
        SNES_REGISTERS = {
            "INIDISP", "OBSEL", "OAMADDL", "OAMADDH", "OAMDATA", "BGMODE", "MOSAIC",
            "BG1SC", "BG2SC", "BG3SC", "BG4SC", "BG12NBA", "BG34NBA",
            "BG1HOFS", "BG1VOFS", "BG2HOFS", "BG2VOFS", "BG3HOFS", "BG3VOFS",
            "BG4HOFS", "BG4VOFS", "VMAIN", "VMADDL", "VMADDH", "VMDATAL", "VMDATAH",
            "NMITIMEN", "WRIO", "MDMAEN", "HDMAEN",
            "APUIO0", "APUIO1", "APUIO2", "APUIO3",
            "CGADD", "CGDATA", "TM", "TS", "TMW", "TSW", "INIDISP",
        }
        ASM_MNEMONICS = {
            "ADC", "AND", "ASL", "BCC", "BCS", "BEQ", "BIT", "BMI", "BNE",
            "BPL", "BRA", "BRK", "BRL", "BVC", "BVS", "CLC", "CLD", "CLI",
            "CLV", "CMP", "COP", "CPX", "CPY", "DEC", "DEX", "DEY", "EOR",
            "INC", "INX", "INY", "JML", "JMP", "JSL", "JSR", "LDA", "LDX",
            "LDY", "LSR", "MVN", "MVP", "NOP", "ORA", "PEA", "PEI", "PER",
            "PHA", "PHB", "PHD", "PHK", "PHP", "PHX", "PHY", "PLA", "PLB",
            "PLD", "PLP", "PLX", "PLY", "REP", "ROL", "ROR", "RTI", "RTL",
            "RTS", "SBC", "SEC", "SED", "SEI", "SEP", "STA", "STP", "STX",
            "STY", "STZ", "TAX", "TAY", "TCD", "TCS", "TDC", "TRB", "TSB",
            "TSC", "TSX", "TXA", "TXS", "TXY", "TYA", "TYX", "WAI", "WDM",
            "XBA", "XCE",
        }

        # Extract entities from sample output
        entities = self._extract_entities(sample.output)
        entities.extend(sample.kg_entities)

        if not entities:
            return 1.0  # No entities to validate

        # Filter out hardware registers
        non_register_entities = [e for e in entities if e.upper() not in SNES_REGISTERS]
        non_register_entities = [
            e for e in non_register_entities if e.upper() not in ASM_MNEMONICS
        ]

        if not non_register_entities:
            return 1.0  # All entities are hardware registers, skip KG check

        # Check how many entities exist in the graph
        nodes = self._kg_graph.get("nodes", {})
        valid_count = sum(1 for e in non_register_entities if e in nodes)

        return valid_count / len(non_register_entities) if non_register_entities else 1.0

    def _extract_entities(self, text: str) -> list[str]:
        """Extract potential entities from text."""
        entities = []

        # Look for common patterns in code/ASM
        patterns = [
            r"\b([A-Z][A-Z0-9_]{2,})\b",  # CONSTANTS
            r"\b(0x[0-9A-Fa-f]+)\b",  # Hex addresses
            r"\b(\$[0-9A-Fa-f]+)\b",  # ASM addresses
            r"\b([A-Z][a-z]+[A-Z]\w*)\b",  # CamelCase identifiers
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)

        return list(set(entities))[:20]  # Limit to 20 entities

    async def _check_hallucination(self, sample: TrainingSample) -> float:
        """Detect potential hallucinations in generated content.

        Returns:
            Risk score 0.0-1.0 (lower is better)
        """
        if self.orchestrator is None:
            return 0.5  # Unknown risk

        # Quick heuristic checks first
        risk = 0.0

        # Check for suspicious patterns
        suspicious_patterns = [
            r"I don't know",
            r"I'm not sure",
            r"might be",
            r"possibly",
            r"I think",
            r"maybe",
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, sample.output, re.IGNORECASE):
                risk += 0.1

        # Check instruction/output length ratio (but be lenient for code)
        instruction_len = len(sample.instruction.split())
        output_len = len(sample.output.split())

        if output_len > 0:
            ratio = instruction_len / output_len
            # More lenient for code domains
            if sample.domain in ("asm", "cpp", "yaze", "gigaleak", "oracle"):
                if ratio > 10 or ratio < 0.05:  # Very extreme imbalance
                    risk += 0.1  # Lower penalty
            else:
                if ratio > 5 or ratio < 0.1:  # Imbalanced
                    risk += 0.2

        # For code samples, skip LLM verification (too slow and unreliable)
        # The domain validators already check code validity
        if risk < 0.3 and sample.domain not in ("asm", "cpp", "yaze", "gigaleak", "oracle"):
            try:
                from hafs.core.orchestrator_v2 import TaskTier

                prompt = f"""Analyze this training sample for factual accuracy:

INSTRUCTION: {sample.instruction[:500]}
OUTPUT (first 500 chars): {sample.output[:500]}

Is the output factually consistent and technically accurate?
Rate confidence 0.0-1.0 that this is accurate (not hallucinated).
Respond with just the number."""

                response = await self.orchestrator.generate(
                    prompt=prompt,
                    tier=TaskTier.FAST,
                )

                # Robustly parse confidence score
                try:
                    response_text = response.content.strip()
                    # Try multiple patterns
                    patterns = [
                        r'^\s*([01]?\.\d+)\s*$',  # Plain number
                        r'^\s*([01]?\.\d+)',  # Number at start
                        r'([01]?\.\d+)\s*$',  # Number at end
                        r'([01]?\.\d+)',  # Number anywhere
                    ]

                    confidence = 0.5  # Default neutral
                    for pattern in patterns:
                        match = re.search(pattern, response_text)
                        if match:
                            value = float(match.group(1))
                            if 0.0 <= value <= 1.0:
                                confidence = value
                                break

                    risk = max(risk, 1.0 - confidence)
                except (ValueError, AttributeError):
                    # If parsing fails, use neutral risk
                    risk = max(risk, 0.5)

            except Exception as e:
                logger.debug(f"LLM hallucination check failed: {e}")

        return min(risk, 1.0)

    def _score_coherence(self, sample: TrainingSample) -> float:
        """Score semantic coherence between instruction and output.

        Domain-aware: code domains don't expect word overlap.
        """
        # For code domains, check if output contains code patterns
        if sample.domain in ("asm", "cpp", "yaze", "gigaleak", "oracle"):
            code_indicators = [
                r'\b(lda|sta|jmp|jsr|rts|php|plp|pha|pla|bne|beq|bcs|bcc)\b',  # ASM
                r'\{|\}',  # Braces
                r';.*$',  # Comments
                r'^\s*(if|for|while|return|void|int|class|struct)',  # C++ keywords
                r'0x[0-9a-fA-F]+',  # Hex addresses
                r'\$[0-9a-fA-F]+',  # ASM hex
                r':$',  # Labels
            ]

            matches = sum(
                1 for pattern in code_indicators
                if re.search(pattern, sample.output, re.IGNORECASE | re.MULTILINE)
            )

            # If output looks like code, give it good coherence
            if matches >= 3:
                return 0.8  # Strong code patterns
            elif matches >= 2:
                return 0.6  # Moderate
            elif matches >= 1:
                return 0.4  # Weak
            else:
                return 0.2  # Doesn't look like code

        # For text domains, use word overlap
        instruction_words = set(sample.instruction.lower().split())
        output_words = set(sample.output.lower().split())

        # Remove common words
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once",
            "and", "but", "or", "nor", "so", "yet", "both", "either",
            "neither", "not", "only", "own", "same", "than", "too", "very",
            "can", "just", "should", "now", "this", "that", "these", "those",
        }

        instruction_words -= stopwords
        output_words -= stopwords

        if not instruction_words or not output_words:
            return 0.5  # Can't determine coherence

        # Jaccard similarity
        intersection = instruction_words & output_words
        union = instruction_words | output_words

        return len(intersection) / len(union) if union else 0.5

    async def check_duplicate(
        self, sample: TrainingSample
    ) -> DuplicateResult:
        """Check if a sample is a duplicate of existing samples.

        Args:
            sample: Sample to check

        Returns:
            DuplicateResult with duplicate status and similarity
        """
        if self.embedding_index is None or self.embedding_index.size() == 0:
            return DuplicateResult(is_duplicate=False, similarity=0.0)

        # Get or generate embedding
        if sample.embedding is None:
            if self.orchestrator:
                try:
                    sample.embedding = await self.orchestrator.embed(sample.instruction)
                except Exception:
                    return DuplicateResult(is_duplicate=False, similarity=0.0)
            else:
                return DuplicateResult(is_duplicate=False, similarity=0.0)

        # Search for similar
        try:
            query = np.array(sample.embedding, dtype=np.float32)
            ids, scores = self.embedding_index.search(query, k=1)

            if len(scores) > 0:
                similarity = float(scores[0])
                is_dup = similarity > self.SIMILARITY_THRESHOLD
                return DuplicateResult(
                    is_duplicate=is_dup,
                    similarity=similarity,
                    matched_id=ids[0] if is_dup else None,
                )
        except Exception as e:
            logger.warning(f"Duplicate check failed: {e}")

        return DuplicateResult(is_duplicate=False, similarity=0.0)

    async def add_to_index(self, sample: TrainingSample) -> bool:
        """Add a sample to the embedding index.

        Args:
            sample: Sample to add

        Returns:
            True if added successfully
        """
        if self.embedding_index is None:
            return False

        if sample.embedding is None:
            if self.orchestrator:
                try:
                    sample.embedding = await self.orchestrator.embed(sample.instruction)
                except Exception:
                    return False
            else:
                return False

        try:
            embedding = np.array(sample.embedding, dtype=np.float32)
            return self.embedding_index.add(sample.sample_id, embedding)
        except Exception as e:
            logger.warning(f"Failed to add to index: {e}")
            return False

    async def filter_samples(
        self,
        samples: list[TrainingSample],
        min_quality: Optional[float] = None,
        deduplicate: bool = True,
        generator_name: str = "unknown",
        run_validation: bool = True,
    ) -> list[TrainingSample]:
        """Filter samples by quality, validation, and deduplication.

        Args:
            samples: List of samples to filter
            min_quality: Minimum quality score (uses domain-specific threshold if None)
            deduplicate: Whether to remove duplicates
            generator_name: Name of generator for feedback tracking
            run_validation: Whether to run domain-specific validation

        Returns:
            Filtered list of samples
        """
        # Domain-specific quality thresholds
        DOMAIN_THRESHOLDS = {
            "asm": 0.4,  # ASM is hard - lower threshold
            "gigaleak": 0.45,  # Original source - adjusted for Gemini Flash capabilities
            "oracle": 0.4,  # ROM hack - lower
            "yaze": 0.5,  # C++ code - medium
            "cpp": 0.5,  # C++ code - medium
            "errors": 0.3,  # Error diagnostics - lowest
            "text": 0.6,  # Natural language - higher
        }

        # Log threshold strategy
        if min_quality is None:
            print(f"[QUALITY] Using per-sample domain-specific quality thresholds", flush=True)
            logger.info(f"Using per-sample domain-specific quality thresholds")
        else:
            print(f"[QUALITY] Using fixed quality threshold: {min_quality}", flush=True)
            logger.info(f"Using fixed quality threshold: {min_quality}")

        filtered: list[TrainingSample] = []
        rejected_validation = 0
        rejected_quality = 0
        rejected_duplicates = 0

        for sample in samples:
            rejection_reason = None

            # Run domain-specific validation first
            if run_validation:
                is_valid, validation_details = await self.validate(sample)
                if not is_valid:
                    rejection_reason = self._RejectionReason.VALIDATION_FAILED if self._RejectionReason else None
                    await self._record_rejection(sample, None, rejection_reason, generator_name)
                    rejected_validation += 1
                    continue

            # Check quality score with per-sample domain-specific threshold
            score = await self.score(sample)

            # Use domain-specific threshold for this sample, or fallback to min_quality
            sample_threshold = min_quality
            if min_quality is None:
                sample_threshold = DOMAIN_THRESHOLDS.get(sample.domain, self.MIN_QUALITY_SCORE)

            if score.overall < sample_threshold:
                # Debug logging for first few rejections
                if rejected_quality < 3:
                    print(f"[QUALITY] Rejected sample (domain={sample.domain}, score={score.overall:.3f}, threshold={sample_threshold:.3f})", flush=True)
                # Determine specific rejection reason
                if self._RejectionReason:
                    if score.diversity_score < 0.3:
                        rejection_reason = self._RejectionReason.LOW_DIVERSITY
                    elif score.kg_consistency < 0.5:
                        rejection_reason = self._RejectionReason.KG_INCONSISTENT
                    elif score.hallucination_risk > 0.5:
                        rejection_reason = self._RejectionReason.HIGH_HALLUCINATION
                    elif score.semantic_coherence < 0.4:
                        rejection_reason = self._RejectionReason.LOW_COHERENCE
                    else:
                        rejection_reason = self._RejectionReason.OTHER

                await self._record_rejection(sample, score, rejection_reason, generator_name)
                rejected_quality += 1
                continue

            sample.quality_score = score.overall

            # Check for duplicates
            if deduplicate:
                dup_result = await self.check_duplicate(sample)
                if dup_result.is_duplicate:
                    rejection_reason = self._RejectionReason.DUPLICATE if self._RejectionReason else None
                    await self._record_rejection(sample, score, rejection_reason, generator_name)
                    rejected_duplicates += 1
                    continue

                # Add to index for future duplicate checks
                await self.add_to_index(sample)

            # Record acceptance
            await self._record_acceptance(sample, score, generator_name)

            # Add to active learning sampler
            if self._active_learner and sample.embedding is not None:
                try:
                    embedding = np.array(sample.embedding, dtype=np.float32)
                    self._active_learner.add_sample(
                        sample.sample_id,
                        embedding,
                        sample.domain,
                        score.overall,
                    )
                except Exception as e:
                    # Active learning failures shouldn't block samples
                    logger.debug(f"Active learning add failed: {e}")

            filtered.append(sample)

        # Save feedback periodically
        if self._feedback_tracker and len(filtered) > 0:
            self._feedback_tracker.save()

        if self._active_learner:
            self._active_learner.save()

        self.last_filter_stats = FilterStats(
            total=len(samples),
            accepted=len(filtered),
            rejected_validation=rejected_validation,
            rejected_quality=rejected_quality,
            rejected_duplicates=rejected_duplicates,
        )

        return filtered

    async def _record_rejection(
        self,
        sample: TrainingSample,
        score: Optional[QualityScore],
        reason: Optional[Any],
        generator_name: str,
    ) -> None:
        """Record a rejected sample in feedback tracker."""
        if not self._feedback_tracker:
            return

        # Create a dummy score if not provided
        if score is None:
            score = QualityScore(
                diversity_score=0.0,
                kg_consistency=0.0,
                hallucination_risk=1.0,
                semantic_coherence=0.0,
            )

        self._feedback_tracker.record_sample(
            sample=sample,
            quality_score=score,
            accepted=False,
            generator_name=generator_name,
            rejection_reason=reason,
        )

    async def _record_acceptance(
        self,
        sample: TrainingSample,
        score: QualityScore,
        generator_name: str,
    ) -> None:
        """Record an accepted sample in feedback tracker."""
        if not self._feedback_tracker:
            return

        self._feedback_tracker.record_sample(
            sample=sample,
            quality_score=score,
            accepted=True,
            generator_name=generator_name,
        )

    def get_feedback_report(self) -> dict[str, Any]:
        """Get feedback report from tracker.

        Returns:
            Report with generator stats, trends, and patterns
        """
        if not self._feedback_tracker:
            return {"error": "Feedback tracker not enabled"}

        return {
            "generators": self._feedback_tracker.get_generator_report(),
            "patterns": self._feedback_tracker.get_rejection_patterns(),
            "thresholds": self._feedback_tracker.thresholds,
        }

    def get_coverage_report(self) -> dict[str, Any]:
        """Get embedding space coverage report.

        Returns:
            Coverage statistics and recommendations
        """
        if not self._active_learner:
            return {"error": "Active learning not enabled"}

        report = self._active_learner.get_coverage_report()
        return {
            "total_samples": report.total_samples,
            "num_regions": report.num_regions,
            "coverage_score": report.coverage_score,
            "sparse_regions": report.sparse_regions,
            "domain_coverage": report.domain_coverage,
        }

    def get_improvement_suggestions(self, generator_name: str) -> list[str]:
        """Get suggestions to improve a generator.

        Args:
            generator_name: Name of the generator

        Returns:
            List of improvement suggestions
        """
        if not self._feedback_tracker:
            return ["Feedback tracker not enabled"]

        return self._feedback_tracker.suggest_improvements(generator_name)
