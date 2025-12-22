"""Bayesian Item Response Theory (IRT) Estimator.

Implements the 1PL Rasch model for ability estimation:
    Pr(Y=1) = logit^-1(θ - β)

Based on "Quantifying Human-AI Synergy" research:
- Separates individual ability (θ) from collaborative ability (κ)
- Uses Bayesian prior for regularization
- Supports online sequential updates
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from models.irt import (
    AbilityEstimate,
    AbilityType,
    ItemResponse,
    EnhancedUserProfile,
    TraitToMScore,
    ToMAssessment,
)
from models.synergy_config import IRTConfig

logger = logging.getLogger(__name__)


@dataclass
class EstimationResult:
    """Result of an ability estimation."""

    theta: float
    se: float
    n_responses: int
    converged: bool
    iterations: int
    log_likelihood: float


class BayesianIRTEstimator:
    """Bayesian 1PL IRT estimator with online updating.

    Uses the Rasch model: Pr(Y=1) = 1 / (1 + exp(-(θ - β)))

    Key features:
    - Maximum Likelihood Estimation via Newton-Raphson
    - Bayesian prior for regularization (N(prior_mean, prior_sd))
    - Sequential Bayesian updates for online learning
    - Separate estimation for θ (individual) and κ (collaborative)
    """

    def __init__(self, config: Optional[IRTConfig] = None):
        """Initialize the IRT estimator.

        Args:
            config: IRT configuration (uses defaults if None)
        """
        self.config = config or IRTConfig()

    def probability(self, theta: float, beta: float) -> float:
        """Calculate Pr(Y=1|θ,β) using the Rasch model.

        Args:
            theta: Ability parameter
            beta: Difficulty parameter

        Returns:
            Probability of success (0-1)
        """
        z = theta - beta
        # Clamp to prevent overflow
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def log_likelihood(
        self,
        theta: float,
        responses: list[ItemResponse],
    ) -> float:
        """Calculate log-likelihood of responses given ability.

        Args:
            theta: Ability parameter
            responses: List of item responses

        Returns:
            Log-likelihood value
        """
        ll = 0.0
        for resp in responses:
            p = self.probability(theta, resp.difficulty_beta)
            if resp.response:
                ll += math.log(max(p, 1e-10))
            else:
                ll += math.log(max(1 - p, 1e-10))
        return ll

    def _log_prior(self, theta: float) -> float:
        """Calculate log of Gaussian prior."""
        diff = theta - self.config.prior_mean
        return -0.5 * (diff ** 2) / self.config.prior_variance

    def _gradient(
        self,
        theta: float,
        responses: list[ItemResponse],
    ) -> float:
        """Calculate gradient of log-posterior.

        First derivative of log-likelihood + log-prior.
        """
        # Likelihood gradient
        grad = 0.0
        for resp in responses:
            p = self.probability(theta, resp.difficulty_beta)
            if resp.response:
                grad += 1 - p
            else:
                grad -= p

        # Prior gradient (Gaussian)
        prior_grad = -(theta - self.config.prior_mean) / self.config.prior_variance

        return grad + prior_grad

    def _hessian(
        self,
        theta: float,
        responses: list[ItemResponse],
    ) -> float:
        """Calculate Hessian of log-posterior.

        Second derivative of log-likelihood + log-prior.
        """
        # Likelihood Hessian (always negative for concave function)
        hess = 0.0
        for resp in responses:
            p = self.probability(theta, resp.difficulty_beta)
            hess -= p * (1 - p)

        # Prior Hessian
        prior_hess = -1.0 / self.config.prior_variance

        return hess + prior_hess

    def estimate_ability(
        self,
        responses: list[ItemResponse],
        prior_theta: Optional[float] = None,
    ) -> EstimationResult:
        """Estimate ability using Newton-Raphson MLE with Bayesian prior.

        Args:
            responses: List of item responses
            prior_theta: Starting estimate (uses config prior_mean if None)

        Returns:
            EstimationResult with theta, SE, and convergence info
        """
        if not responses:
            return EstimationResult(
                theta=self.config.prior_mean,
                se=self.config.prior_sd,
                n_responses=0,
                converged=True,
                iterations=0,
                log_likelihood=0.0,
            )

        # Initialize
        theta = prior_theta if prior_theta is not None else self.config.prior_mean
        converged = False
        iterations = 0

        # Newton-Raphson iteration
        for i in range(self.config.max_iterations):
            grad = self._gradient(theta, responses)
            hess = self._hessian(theta, responses)

            # Avoid division by zero
            if abs(hess) < 1e-10:
                break

            # Newton step
            delta = -grad / hess
            theta_new = theta + delta

            # Clamp to reasonable range
            theta_new = max(-5.0, min(5.0, theta_new))

            # Check convergence
            if abs(delta) < self.config.convergence_threshold:
                converged = True
                theta = theta_new
                iterations = i + 1
                break

            theta = theta_new
            iterations = i + 1

        # Calculate standard error from Fisher information
        # SE = 1 / sqrt(-Hessian) at the MLE
        hess_final = self._hessian(theta, responses)
        if abs(hess_final) > 1e-10:
            se = 1.0 / math.sqrt(-hess_final)
        else:
            se = self.config.prior_sd

        ll = self.log_likelihood(theta, responses)

        return EstimationResult(
            theta=theta,
            se=se,
            n_responses=len(responses),
            converged=converged,
            iterations=iterations,
            log_likelihood=ll,
        )

    def update_sequential(
        self,
        current: AbilityEstimate,
        new_response: ItemResponse,
    ) -> AbilityEstimate:
        """Update ability estimate with a new response using Bayesian updating.

        Uses the current estimate as the prior for the new estimate.
        More efficient than re-estimating from all responses.

        Args:
            current: Current ability estimate
            new_response: New item response to incorporate

        Returns:
            Updated AbilityEstimate
        """
        # Use current estimate as prior
        prior_theta = current.theta
        prior_se = current.se

        # Calculate posterior using Bayesian update
        # For single observation: approximate with weighted combination

        # Likelihood contribution
        p = self.probability(prior_theta, new_response.difficulty_beta)
        observed = 1.0 if new_response.response else 0.0

        # Information from this observation
        info_obs = p * (1 - p)

        # Prior precision
        prior_precision = 1.0 / (prior_se ** 2) if prior_se > 0 else 1.0

        # Posterior precision
        post_precision = prior_precision + info_obs

        # Posterior mean (weighted combination)
        # Score contribution from observation
        score = observed - p

        # Update theta
        post_theta = prior_theta + score / post_precision

        # Clamp to reasonable range
        post_theta = max(-5.0, min(5.0, post_theta))

        # Posterior SE
        post_se = 1.0 / math.sqrt(post_precision) if post_precision > 0 else prior_se

        return AbilityEstimate(
            ability_type=current.ability_type,
            theta=post_theta,
            se=post_se,
            n_responses=current.n_responses + 1,
            last_updated=datetime.now(),
        )

    def calculate_synergy_gain(
        self,
        theta_individual: AbilityEstimate,
        kappa_collaborative: AbilityEstimate,
    ) -> float:
        """Calculate synergy gain: κ - θ.

        Positive values indicate the user benefits from AI collaboration.
        Negative values indicate the user performs better alone.

        Args:
            theta_individual: Individual ability estimate
            kappa_collaborative: Collaborative ability estimate

        Returns:
            Synergy gain value
        """
        return kappa_collaborative.theta - theta_individual.theta

    def update_profile_with_response(
        self,
        profile: EnhancedUserProfile,
        response: ItemResponse,
        is_collaborative: bool,
    ) -> EnhancedUserProfile:
        """Update a user profile with a new task response.

        Args:
            profile: User profile to update
            response: The task response
            is_collaborative: True if task was done with AI assistance

        Returns:
            Updated profile (mutated in place)
        """
        if is_collaborative:
            profile.kappa_collaborative = self.update_sequential(
                profile.kappa_collaborative,
                response,
            )
        else:
            profile.theta_individual = self.update_sequential(
                profile.theta_individual,
                response,
            )

        # Add to response history (with limit)
        profile.item_responses.append(response)
        if len(profile.item_responses) > self.config.max_history:
            profile.item_responses.pop(0)

        # Update synergy gain
        profile.synergy_gain = self.calculate_synergy_gain(
            profile.theta_individual,
            profile.kappa_collaborative,
        )
        profile.last_synergy_update = datetime.now()

        return profile

    def update_profile_with_tom(
        self,
        profile: EnhancedUserProfile,
        assessment: ToMAssessment,
    ) -> EnhancedUserProfile:
        """Update a user profile with a new ToM assessment.

        Updates trait-level means and within-user deviations.

        Args:
            profile: User profile to update
            assessment: The ToM assessment

        Returns:
            Updated profile (mutated in place)
        """
        # Ensure traits are initialized
        profile.initialize_tom_traits()

        # Update each trait
        for trait, score in assessment.dimension_scores.items():
            if trait in profile.tom_traits:
                trait_score = profile.tom_traits[trait]

                # Exponential moving average for trait mean
                alpha = 0.1  # Learning rate
                old_mean = trait_score.mean_score
                trait_score.mean_score = old_mean + alpha * (score - old_mean)

                # Within-user deviation (current vs stable mean)
                trait_score.within_deviation = score - trait_score.mean_score

                trait_score.n_assessments += 1

        # Add assessment to history
        profile.tom_assessments.append(assessment)
        if len(profile.tom_assessments) > 100:
            profile.tom_assessments.pop(0)

        return profile

    def get_reliability_info(self, estimate: AbilityEstimate) -> dict:
        """Get reliability information for an ability estimate.

        Args:
            estimate: The ability estimate

        Returns:
            Dictionary with reliability metrics
        """
        return {
            "theta": estimate.theta,
            "se": estimate.se,
            "n_responses": estimate.n_responses,
            "is_reliable": estimate.is_reliable,
            "confidence_interval_95": estimate.confidence_interval_95,
            "min_responses_needed": max(
                0, self.config.min_responses - estimate.n_responses
            ),
        }
