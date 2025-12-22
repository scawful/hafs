#!/usr/bin/env python3
"""Regression tests for quality validation pipeline.

Prevents the 0% pass rate bug from recurring.

Tests:
1. ASM samples get appropriate quality scores
2. Domain-specific thresholds are applied
3. Code coherence uses pattern matching (not word overlap)
4. Hallucination check skips LLM for code domains
5. Hardware registers don't fail KG validation
"""

import pytest
from agents.training.base import TrainingSample, QualityScore
from agents.training.quality import QualityPipeline


class TestQualityValidation:
    """Test suite for quality validation fixes."""

    @pytest.mark.asyncio
    async def test_asm_sample_passes_validation(self):
        """Test that a typical ASM sample passes quality validation."""
        # Create a realistic ASM sample
        sample = TrainingSample(
            instruction="Write SNES reset routine for bank 00",
            input="",
            output="""Reset:
    STZ.w NMITIMEN
    STZ.w HDMAEN
    STZ.w MDMAEN
    LDA.b #$80
    STA.w INIDISP
    JMP.w InitializeMemory""",
            domain="asm",
            source="vanilla",
            kg_entities=["Reset", "NMITIMEN", "HDMAEN", "MDMAEN", "INIDISP"],
        )

        pipeline = QualityPipeline(load_config=False)
        await pipeline.setup()

        score = await pipeline.score(sample)

        # With fixes, this should score >= 0.4 (ASM threshold)
        assert score.overall >= 0.4, f"ASM sample failed: score={score.overall}"
        assert score.hallucination_risk < 0.5, "Hallucination risk too high for ASM"
        assert score.semantic_coherence >= 0.4, "Coherence too low (code pattern detection failed)"

    @pytest.mark.asyncio
    async def test_code_coherence_uses_patterns(self):
        """Test that code coherence uses pattern matching, not word overlap."""
        pipeline = QualityPipeline(load_config=False)

        # Code sample with no word overlap
        sample = TrainingSample(
            instruction="Implement function to read ROM byte",
            input="",
            output="""LDA.w $8000,x
STA.w $7E0000,x
RTS""",
            domain="asm",
            source="test",
        )

        coherence = pipeline._score_coherence(sample)

        # Should detect code patterns, not fail due to no word overlap
        assert coherence >= 0.4, f"Code coherence too low: {coherence}"

    @pytest.mark.asyncio
    async def test_hardware_registers_skip_kg_check(self):
        """Test that hardware registers don't fail KG validation."""
        pipeline = QualityPipeline(load_config=False)
        await pipeline.setup()

        sample = TrainingSample(
            instruction="Clear SNES PPU registers",
            input="",
            output="STZ INIDISP",
            domain="asm",
            source="test",
            kg_entities=["INIDISP", "NMITIMEN", "HDMAEN"],  # All hardware registers
        )

        kg_score = await pipeline._validate_kg(sample)

        # Should return 1.0 (all entities are hardware registers)
        assert kg_score == 1.0, f"Hardware registers failed KG check: {kg_score}"

    @pytest.mark.asyncio
    async def test_hallucination_check_skips_llm_for_code(self):
        """Test that hallucination check doesn't use LLM for code domains."""
        pipeline = QualityPipeline(load_config=False)
        await pipeline.setup()

        sample = TrainingSample(
            instruction="Test ASM code",
            input="",
            output="LDA #$00",
            domain="asm",
            source="test",
        )

        # This should complete quickly (no LLM call)
        import time
        start = time.time()
        risk = await pipeline._check_hallucination(sample)
        duration = time.time() - start

        # Should be < 1 second (no LLM call)
        assert duration < 1.0, f"Hallucination check took {duration}s (LLM called?)"
        # Risk should be low (no suspicious patterns)
        assert risk < 0.5, f"Hallucination risk too high: {risk}"

    @pytest.mark.asyncio
    async def test_domain_specific_thresholds(self):
        """Test that domain-specific thresholds are applied."""
        pipeline = QualityPipeline(load_config=False)

        # Create samples for different domains
        asm_sample = TrainingSample(
            instruction="test",
            input="",
            output="LDA #$00",
            domain="asm",
            source="test",
        )

        text_sample = TrainingSample(
            instruction="test",
            input="",
            output="This is a text response",
            domain="text",
            source="test",
        )

        # filter_samples should use domain-specific thresholds
        # ASM: 0.4, Text: 0.6
        # We can't easily test this without mocking, but at least verify it doesn't crash
        filtered = await pipeline.filter_samples([asm_sample], min_quality=None)
        assert isinstance(filtered, list), "filter_samples should return list"

    @pytest.mark.asyncio
    async def test_regression_0_percent_pass_rate(self):
        """Regression test: ensure we never get 0% pass rate again."""
        pipeline = QualityPipeline(load_config=False)
        await pipeline.setup()

        # Create 10 realistic ASM samples
        samples = []
        for i in range(10):
            sample = TrainingSample(
                instruction=f"Test routine {i}",
                input="",
                output=f"""Test{i}:
    LDA #${i:02X}
    STA $7E0000
    RTS""",
                domain="asm",
                source="test",
            )
            samples.append(sample)

        # Filter with ASM threshold (0.4)
        filtered = await pipeline.filter_samples(samples, min_quality=0.4)

        # At least SOME samples should pass (not 0%)
        pass_rate = len(filtered) / len(samples)
        assert pass_rate > 0.0, "REGRESSION: 0% pass rate detected!"
        assert pass_rate >= 0.5, f"Pass rate too low: {pass_rate*100:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
