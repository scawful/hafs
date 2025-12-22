import warnings
from agents.knowledge.rom import RomHackingSpecialist

warnings.warn(
    "agents.knowledge.rom is deprecated. Import from 'agents.knowledge.rom' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export classes
RomHackingSpecialist = RomHackingSpecialist
