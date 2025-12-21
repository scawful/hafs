# Services

Cross-platform service management for HAFS. This is the canonical location for service management.

## Structure

```
services/
├── __init__.py          # Main exports
├── manager.py           # ServiceManager
├── models.py            # ServiceDefinition, ServiceStatus, etc.
└── adapters/
    ├── __init__.py
    ├── base.py          # ServiceAdapter ABC
    ├── launchd.py       # macOS adapter
    └── systemd.py       # Linux adapter
```

## Usage

```python
from services import ServiceManager, ServiceDefinition

manager = ServiceManager()

# Define a service
definition = ServiceDefinition(
    name="my-daemon",
    label="My Daemon",
    command=["python", "-m", "my_module"],
)

# Install and start
await manager.install(definition)
await manager.start("my-daemon")

# Check status
status = await manager.status("my-daemon")
print(f"State: {status.state}")
```

## Backward Compatibility

For backward compatibility, `hafs.core.services` re-exports everything from this module.
New code should import directly from `services`.
