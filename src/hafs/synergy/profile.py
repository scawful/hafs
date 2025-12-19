"""User profile management for synergy tracking."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

from hafs.models.synergy import (
    ResponseQuality,
    UserPreferences,
    UserProfile,
)


class UserProfileManager:
    """Manages user profiles for tracking preferences and interaction history."""

    def __init__(self, storage_path: Path) -> None:
        """
        Initialize the user profile manager.

        Args:
            storage_path: Path to the directory for storing user profiles.
        """
        self._storage_path = storage_path
        self._profiles: dict[str, UserProfile] = {}

        # Ensure storage directory exists
        self._storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing profiles
        self.load()

    def get_or_create(self, user_id: str) -> UserProfile:
        """
        Get an existing user profile or create a new one.

        Args:
            user_id: The user's identifier.

        Returns:
            UserProfile instance for the user.
        """
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(
                id=UUID(int=hash(user_id) & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF),
                preferences=UserPreferences(),
            )

        return self._profiles[user_id]

    def update_from_interaction(
        self,
        user_id: str,
        prompt: str,
        response: str,
        quality: ResponseQuality,
    ) -> None:
        """
        Update user profile based on an interaction.

        Args:
            user_id: The user's identifier.
            prompt: The user's prompt.
            response: The agent's response.
            quality: The quality metrics for the response.
        """
        profile = self.get_or_create(user_id)

        # Update interaction count and timestamp
        profile.increment_interactions()

        # Learn from the interaction
        self._learn_preferences(profile, prompt, response)

        # Update average quality score
        current_avg = profile.preferences.avg_response_quality
        count = profile.preferences.interaction_count

        # Calculate new average
        new_avg = (current_avg * (count - 1) + quality.overall) / count
        profile.preferences.avg_response_quality = new_avg

        # Save updated profile
        self.save()

    def _learn_preferences(
        self, profile: UserProfile, prompt: str, response: str
    ) -> None:
        """
        Learn user preferences from an interaction.

        Args:
            profile: The user's profile.
            prompt: The user's prompt.
            response: The agent's response.
        """
        # Analyze prompt length to infer detail preference
        prompt_words = len(prompt.split())

        if prompt_words < 20:
            # Short prompts might indicate preference for concise communication
            if profile.preferences.detail_level == "medium":
                profile.preferences.detail_level = "low"
        elif prompt_words > 75:
            # Long prompts might indicate preference for detailed communication
            if profile.preferences.detail_level == "medium":
                profile.preferences.detail_level = "high"

        # Analyze communication style based on prompt characteristics
        has_technical_terms = any(
            term in prompt.lower()
            for term in [
                "api",
                "algorithm",
                "function",
                "class",
                "module",
                "framework",
                "architecture",
                "implementation",
            ]
        )

        if has_technical_terms and profile.preferences.expertise_level == "intermediate":
            profile.preferences.expertise_level = "expert"

        # Track prompt in interaction history (keep last 20)
        if len(profile.preferences.prompt_history) >= 20:
            profile.preferences.prompt_history.pop(0)
        profile.preferences.prompt_history.append(prompt[:100])  # Store truncated version

    def save(self) -> None:
        """Save all user profiles to disk."""
        profiles_file = self._storage_path / "user_profiles.json"

        # Convert profiles to serializable format
        profiles_data = {}
        for user_id, profile in self._profiles.items():
            profiles_data[user_id] = {
                "id": str(profile.id),
                "preferences": {
                    "preferred_response_length": profile.preferences.preferred_response_length,
                    "expertise_level": profile.preferences.expertise_level,
                    "communication_style": profile.preferences.communication_style,
                    "avg_response_quality": profile.preferences.avg_response_quality,
                    "interaction_count": profile.preferences.interaction_count,
                    "detail_level": profile.preferences.detail_level,
                    "prompt_history": profile.preferences.prompt_history,
                },
                "last_interaction": (
                    profile.last_interaction.isoformat()
                    if profile.last_interaction
                    else None
                ),
            }

        # Write to file
        with open(profiles_file, "w") as f:
            json.dump(profiles_data, f, indent=2)

    def load(self) -> None:
        """Load user profiles from disk."""
        profiles_file = self._storage_path / "user_profiles.json"

        if not profiles_file.exists():
            return

        try:
            with open(profiles_file, "r") as f:
                profiles_data = json.load(f)

            # Reconstruct profiles
            for user_id, data in profiles_data.items():
                preferences = UserPreferences(
                    preferred_response_length=data["preferences"].get(
                        "preferred_response_length", "medium"
                    ),
                    expertise_level=data["preferences"].get(
                        "expertise_level", "intermediate"
                    ),
                    communication_style=data["preferences"].get(
                        "communication_style", "balanced"
                    ),
                    avg_response_quality=data["preferences"].get(
                        "avg_response_quality", 0.0
                    ),
                    interaction_count=data["preferences"].get(
                        "interaction_count", 0
                    ),
                    detail_level=data["preferences"].get(
                        "detail_level", "balanced"
                    ),
                    prompt_history=data["preferences"].get(
                        "prompt_history", []
                    ),
                )

                profile = UserProfile(
                    id=UUID(data["id"]),
                    preferences=preferences,
                    last_interaction=(
                        datetime.fromisoformat(data["last_interaction"])
                        if data.get("last_interaction")
                        else None
                    ),
                )

                self._profiles[user_id] = profile

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If loading fails, start with empty profiles
            print(f"Warning: Failed to load user profiles: {e}")
            self._profiles = {}

    def get_all_profiles(self) -> dict[str, UserProfile]:
        """
        Get all user profiles.

        Returns:
            Dictionary mapping user IDs to UserProfile instances.
        """
        return self._profiles.copy()

    def delete_profile(self, user_id: str) -> bool:
        """
        Delete a user profile.

        Args:
            user_id: The user's identifier.

        Returns:
            True if profile was deleted, False if it didn't exist.
        """
        if user_id in self._profiles:
            del self._profiles[user_id]
            self.save()
            return True
        return False
