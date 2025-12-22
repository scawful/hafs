"""Synergy Service for IRT-based ability tracking.

Provides:
- LLM-based Theory of Mind assessment
- Bayesian IRT ability estimation
- User profile management with persistence
- Integration with quota manager

Based on "Quantifying Human-AI Synergy" research.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from models.irt import (
    AbilityEstimate,
    AbilityType,
    DifficultyLevel,
    EnhancedUserProfile,
    ItemResponse,
    ToMAssessment,
)
from models.synergy_config import (
    AssessmentMode,
    SynergyServiceConfig,
)
from synergy.tom_assessor import ToMAssessor
from synergy.irt_estimator import BayesianIRTEstimator

logger = logging.getLogger(__name__)


@dataclass
class SynergyStatus:
    """Status of the synergy service."""

    running: bool
    enabled: bool
    tom_assessor_stats: dict
    profiles_count: int
    total_interactions: int
    last_update: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SynergySummary:
    """Summary of user synergy for UI display."""

    user_id: str
    theta_individual: float
    theta_individual_se: float
    theta_individual_reliable: bool
    kappa_collaborative: float
    kappa_collaborative_se: float
    kappa_collaborative_reliable: bool
    synergy_gain: float
    ai_benefit: str
    recent_tom_score: Optional[float]
    tom_trend: str

    def to_dict(self) -> dict:
        return asdict(self)


class SynergyService:
    """Background service for synergy tracking and ability estimation.

    Manages:
    - ToM assessor for LLM-based assessment
    - IRT estimator for ability tracking
    - User profiles with persistence
    - Batch processing queue

    Example:
        service = SynergyService()
        await service.start()

        # Record an interaction
        await service.record_interaction(
            user_id="default",
            prompt="How do I implement a binary search?",
            response="Here's how to implement binary search...",
            task_difficulty="medium",
            task_success=True,
            is_collaborative=True,
        )

        # Get synergy summary
        summary = await service.get_synergy_summary("default")
    """

    def __init__(
        self,
        config: Optional[SynergyServiceConfig] = None,
        data_dir: Optional[Path] = None,
    ):
        """Initialize the synergy service.

        Args:
            config: Service configuration (uses defaults if None)
            data_dir: Directory for data persistence (uses config default if None)
        """
        self.config = config or SynergyServiceConfig()

        # Data directory
        if data_dir:
            self._data_dir = Path(data_dir)
        else:
            self._data_dir = Path(self.config.profile_path).expanduser()

        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._profiles_file = self._data_dir / "profiles.json"
        self._status_file = self._data_dir / "status.json"
        self._assessments_dir = self._data_dir / "assessments"
        self._assessments_dir.mkdir(exist_ok=True)

        # Components
        self._tom_assessor = ToMAssessor(config=self.config.tom_assessment)
        self._irt_estimator = BayesianIRTEstimator(config=self.config.irt_estimation)

        # State
        self._profiles: dict[str, EnhancedUserProfile] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._total_interactions = 0

        # Batch queue for background processing
        self._batch_queue: list[dict] = []

    async def start(self) -> None:
        """Start the background service."""
        if self._running:
            return

        if not self.config.enabled:
            logger.info("Synergy service is disabled")
            return

        # Load persisted profiles
        await self._load_profiles()

        # Initialize ToM assessor
        await self._tom_assessor.initialize()

        self._running = True
        self._task = asyncio.create_task(self._background_loop())

        # Save initial status
        await self._save_status()

        logger.info("Synergy service started")

    async def stop(self) -> None:
        """Stop the background service."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Close assessor
        await self._tom_assessor.close()

        # Persist profiles
        await self._save_profiles()
        await self._save_status()

        logger.info("Synergy service stopped")

    async def _background_loop(self) -> None:
        """Background loop for batch processing."""
        while self._running:
            try:
                # Process batch queue
                if self._batch_queue:
                    await self._process_batch()

                # Save status periodically
                await self._save_status()

                # Wait before next iteration
                await asyncio.sleep(
                    self.config.tom_assessment.batch_interval_seconds
                    if self.config.tom_assessment.is_batch_mode
                    else 60
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in synergy service loop: {e}")
                await asyncio.sleep(10)

    async def record_interaction(
        self,
        user_id: str,
        prompt: str,
        response: str,
        task_difficulty: Optional[str] = None,
        task_success: Optional[bool] = None,
        is_collaborative: bool = True,
    ) -> Optional[SynergySummary]:
        """Record a human-AI interaction.

        Args:
            user_id: User identifier
            prompt: The human's prompt
            response: The AI's response
            task_difficulty: Difficulty level (trivial/easy/medium/hard/expert)
            task_success: Whether the task was completed successfully
            is_collaborative: Whether this was AI-assisted (True) or solo (False)

        Returns:
            Updated synergy summary for the user
        """
        if not self.config.enabled:
            return None

        self._total_interactions += 1

        # Get or create profile
        profile = await self._get_or_create_profile(user_id)

        # Queue ToM assessment if appropriate
        if self._tom_assessor.should_assess(prompt):
            if self.config.tom_assessment.is_batch_mode:
                self._batch_queue.append({
                    "user_id": user_id,
                    "prompt": prompt,
                    "response": response,
                    "timestamp": datetime.now().isoformat(),
                })
            else:
                # Immediate assessment
                asyncio.create_task(
                    self._assess_and_update(user_id, prompt, response)
                )

        # Update IRT if task outcome provided
        if task_success is not None and task_difficulty:
            difficulty_beta = self.config.difficulty_estimation.get_difficulty(
                task_difficulty
            )

            item_response = ItemResponse(
                item_id=f"{user_id}_{self._total_interactions}",
                response=task_success,
                difficulty_beta=difficulty_beta,
                timestamp=datetime.now(),
                context=task_difficulty,
            )

            self._irt_estimator.update_profile_with_response(
                profile,
                item_response,
                is_collaborative,
            )

        # Persist if configured
        if self.config.persist_profiles:
            await self._save_profiles()

        return self._profile_to_summary(user_id, profile)

    async def _assess_and_update(
        self,
        user_id: str,
        prompt: str,
        response: str,
    ) -> None:
        """Perform ToM assessment and update profile."""
        try:
            assessment = await self._tom_assessor.assess(prompt, response)
            if assessment:
                profile = await self._get_or_create_profile(user_id)
                self._irt_estimator.update_profile_with_tom(profile, assessment)

                # Save assessment
                await self._save_assessment(assessment)

                if self.config.persist_profiles:
                    await self._save_profiles()

        except Exception as e:
            logger.error(f"ToM assessment failed: {e}")

    async def _process_batch(self) -> None:
        """Process queued interactions in batch."""
        if not self._batch_queue:
            return

        batch = self._batch_queue[:self.config.tom_assessment.batch_size]
        self._batch_queue = self._batch_queue[self.config.tom_assessment.batch_size:]

        logger.info(f"Processing synergy batch of {len(batch)} items")

        for item in batch:
            await self._assess_and_update(
                item["user_id"],
                item["prompt"],
                item["response"],
            )

    async def get_synergy_summary(self, user_id: str) -> Optional[SynergySummary]:
        """Get synergy summary for a user.

        Args:
            user_id: User identifier

        Returns:
            SynergySummary or None if user not found
        """
        if user_id not in self._profiles:
            return None

        return self._profile_to_summary(user_id, self._profiles[user_id])

    async def get_profile(self, user_id: str) -> Optional[EnhancedUserProfile]:
        """Get full profile for a user.

        Args:
            user_id: User identifier

        Returns:
            EnhancedUserProfile or None if not found
        """
        return self._profiles.get(user_id)

    async def get_status(self) -> SynergyStatus:
        """Get current service status."""
        return SynergyStatus(
            running=self._running,
            enabled=self.config.enabled,
            tom_assessor_stats=self._tom_assessor.get_stats(),
            profiles_count=len(self._profiles),
            total_interactions=self._total_interactions,
            last_update=datetime.now().isoformat(),
        )

    def _profile_to_summary(
        self,
        user_id: str,
        profile: EnhancedUserProfile,
    ) -> SynergySummary:
        """Convert profile to summary."""
        return SynergySummary(
            user_id=user_id,
            theta_individual=profile.theta_individual.theta,
            theta_individual_se=profile.theta_individual.se,
            theta_individual_reliable=profile.theta_individual.is_reliable,
            kappa_collaborative=profile.kappa_collaborative.theta,
            kappa_collaborative_se=profile.kappa_collaborative.se,
            kappa_collaborative_reliable=profile.kappa_collaborative.is_reliable,
            synergy_gain=profile.synergy_gain,
            ai_benefit=profile.ai_benefit,
            recent_tom_score=profile.recent_tom_score,
            tom_trend=profile.tom_trend,
        )

    async def _get_or_create_profile(self, user_id: str) -> EnhancedUserProfile:
        """Get existing profile or create new one."""
        if user_id not in self._profiles:
            profile = EnhancedUserProfile()
            profile.initialize_tom_traits()
            self._profiles[user_id] = profile

        return self._profiles[user_id]

    async def _load_profiles(self) -> None:
        """Load profiles from disk."""
        if not self._profiles_file.exists():
            return

        try:
            data = json.loads(self._profiles_file.read_text())
            for user_id, profile_data in data.items():
                self._profiles[user_id] = self._deserialize_profile(profile_data)
            logger.info(f"Loaded {len(self._profiles)} profiles")
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")

    async def _save_profiles(self) -> None:
        """Save profiles to disk."""
        if not self.config.persist_profiles:
            return

        try:
            data = {
                user_id: self._serialize_profile(profile)
                for user_id, profile in self._profiles.items()
            }
            self._profiles_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save profiles: {e}")

    async def _save_status(self) -> None:
        """Save service status to disk."""
        try:
            status = await self.get_status()
            self._status_file.write_text(json.dumps(status.to_dict(), indent=2))
        except Exception as e:
            logger.error(f"Failed to save status: {e}")

    async def _save_assessment(self, assessment: ToMAssessment) -> None:
        """Save individual assessment to disk."""
        try:
            filename = f"{assessment.id}.json"
            filepath = self._assessments_dir / filename

            data = {
                "id": str(assessment.id),
                "timestamp": assessment.timestamp.isoformat(),
                "scores": assessment.dimension_scores,
                "overall_score": assessment.overall_score,
                "assessor_model": assessment.assessor_model,
                "latency_ms": assessment.latency_ms,
                "reasoning": assessment.reasoning,
            }

            filepath.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save assessment: {e}")

    def _serialize_profile(self, profile: EnhancedUserProfile) -> dict:
        """Serialize profile for JSON storage."""
        return {
            "id": str(profile.id),
            "theta_individual": {
                "theta": profile.theta_individual.theta,
                "se": profile.theta_individual.se,
                "n_responses": profile.theta_individual.n_responses,
            },
            "kappa_collaborative": {
                "theta": profile.kappa_collaborative.theta,
                "se": profile.kappa_collaborative.se,
                "n_responses": profile.kappa_collaborative.n_responses,
            },
            "synergy_gain": profile.synergy_gain,
            "tom_traits": {
                trait: {
                    "mean_score": score.mean_score,
                    "within_deviation": score.within_deviation,
                    "n_assessments": score.n_assessments,
                }
                for trait, score in profile.tom_traits.items()
            },
            "last_synergy_update": profile.last_synergy_update.isoformat(),
        }

    def _deserialize_profile(self, data: dict) -> EnhancedUserProfile:
        """Deserialize profile from JSON."""
        from models.irt import TraitToMScore

        profile = EnhancedUserProfile()

        # Ability estimates
        if "theta_individual" in data:
            profile.theta_individual.theta = data["theta_individual"]["theta"]
            profile.theta_individual.se = data["theta_individual"]["se"]
            profile.theta_individual.n_responses = data["theta_individual"]["n_responses"]

        if "kappa_collaborative" in data:
            profile.kappa_collaborative.theta = data["kappa_collaborative"]["theta"]
            profile.kappa_collaborative.se = data["kappa_collaborative"]["se"]
            profile.kappa_collaborative.n_responses = data["kappa_collaborative"]["n_responses"]

        profile.synergy_gain = data.get("synergy_gain", 0.0)

        # ToM traits
        for trait, trait_data in data.get("tom_traits", {}).items():
            profile.tom_traits[trait] = TraitToMScore(
                trait=trait,
                mean_score=trait_data.get("mean_score", 2.5),
                within_deviation=trait_data.get("within_deviation", 0.0),
                n_assessments=trait_data.get("n_assessments", 0),
            )

        return profile
