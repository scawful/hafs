"""AFS permission policy enforcement."""

from __future__ import annotations

from config.schema import AFSDirectoryConfig, PolicyType
from models.afs import MountType


class PolicyEnforcer:
    """Enforces AFS permission policies.

    Validates operations against configured policies for each mount type.
    """

    def __init__(self, directories: list[AFSDirectoryConfig]):
        """Initialize enforcer with directory configurations.

        Args:
            directories: List of AFS directory configurations.
        """
        self._policies = {d.name: d.policy for d in directories}

    def get_policy(self, mount_type: MountType) -> PolicyType:
        """Get the policy for a mount type.

        Args:
            mount_type: The mount type to get policy for.

        Returns:
            PolicyType for the mount type, defaults to READ_ONLY.
        """
        return self._policies.get(mount_type.value, PolicyType.READ_ONLY)

    def can_read(self, mount_type: MountType) -> bool:
        """Check if reading is allowed.

        All mount types allow reading.

        Args:
            mount_type: The mount type to check.

        Returns:
            Always True (reading is always allowed).
        """
        return True

    def can_write(self, mount_type: MountType) -> bool:
        """Check if writing is allowed.

        Args:
            mount_type: The mount type to check.

        Returns:
            True if policy is WRITABLE or EXECUTABLE.
        """
        policy = self.get_policy(mount_type)
        return policy in (PolicyType.WRITABLE, PolicyType.EXECUTABLE)

    def can_execute(self, mount_type: MountType) -> bool:
        """Check if execution is allowed.

        Args:
            mount_type: The mount type to check.

        Returns:
            True if policy is EXECUTABLE.
        """
        return self.get_policy(mount_type) == PolicyType.EXECUTABLE

    def validate_operation(
        self,
        mount_type: MountType,
        operation: str,  # "read", "write", "execute"
    ) -> tuple[bool, str]:
        """Validate an operation against policy.

        Args:
            mount_type: The mount type to validate against.
            operation: The operation to validate ("read", "write", "execute").

        Returns:
            Tuple of (is_allowed, error_message).
            If allowed, error_message is empty string.
        """
        policy = self.get_policy(mount_type)

        if operation == "read":
            return (True, "")

        if operation == "write":
            if policy in (PolicyType.WRITABLE, PolicyType.EXECUTABLE):
                return (True, "")
            return (
                False,
                f"{mount_type.value} is {policy.value}, writing not allowed",
            )

        if operation == "execute":
            if policy == PolicyType.EXECUTABLE:
                return (True, "")
            return (
                False,
                f"{mount_type.value} is {policy.value}, execution not allowed",
            )

        return (False, f"Unknown operation: {operation}")

    def get_policy_description(self, mount_type: MountType) -> str:
        """Get human-readable description of mount type's policy.

        Args:
            mount_type: The mount type to describe.

        Returns:
            String description of the policy.
        """
        policy = self.get_policy(mount_type)
        descriptions = {
            PolicyType.READ_ONLY: "Read-only (no modifications allowed)",
            PolicyType.WRITABLE: "Writable (modifications allowed)",
            PolicyType.EXECUTABLE: "Executable (can run scripts/binaries)",
        }
        return descriptions.get(policy, "Unknown policy")
