"""Quality pipeline fixes for domain-specific validation.

FIXES:
1. Domain-aware thresholds (ASM gets more lenient scoring)
2. Code-aware coherence (don't expect word overlap for ASM)
3. Robust hallucination check (better LLM response parsing)
4. Hardware register exemption for KG (don't fail on SNES registers)
"""

# Domain-specific quality thresholds
DOMAIN_THRESHOLDS = {
    "asm": 0.4,  # ASM is hard - lower threshold
    "gigaleak": 0.5,  # Original source - medium
    "oracle": 0.4,  # ROM hack - lower
    "yaze": 0.5,  # C++ code - medium
    "cpp": 0.5,  # C++ code - medium
    "errors": 0.3,  # Error diagnostics - lowest
    "text": 0.6,  # Natural language - higher
}

# Default threshold for unknown domains
DEFAULT_THRESHOLD = 0.5

# SNES hardware registers (should NOT be checked against KG)
SNES_REGISTERS = {
    # PPU registers
    "INIDISP", "OBSEL", "OAMADDL", "OAMADDH", "OAMDATA", "BGMODE", "MOSAIC",
    "BG1SC", "BG2SC", "BG3SC", "BG4SC", "BG12NBA", "BG34NBA",
    "BG1HOFS", "BG1VOFS", "BG2HOFS", "BG2VOFS", "BG3HOFS", "BG3VOFS",
    "BG4HOFS", "BG4VOFS", "VMAIN", "VMADDL", "VMADDH", "VMDATAL", "VMDATAH",
    "M7SEL", "M7A", "M7B", "M7C", "M7D", "M7X", "M7Y",
    "CGADD", "CGDATA", "W12SEL", "W34SEL", "WOBJSEL", "WH0", "WH1", "WH2", "WH3",
    "WBGLOG", "WOBJLOG", "TM", "TS", "TMW", "TSW",
    "CGWSEL", "CGADSUB", "COLDATA", "SETINI",

    # CPU/DMA registers
    "NMITIMEN", "WRIO", "WRMPYA", "WRMPYB", "WRDIVL", "WRDIVH", "WRDIVB",
    "HTIMEL", "HTIMEH", "VTIMEL", "VTIMEH", "MDMAEN", "HDMAEN",

    # APU registers
    "APUIO0", "APUIO1", "APUIO2", "APUIO3",

    # DMA channels (0-7)
    "DMAP0", "BBAD0", "A1T0L", "A1T0H", "A1B0", "DAS0L", "DAS0H", "DASB0", "A2A0L", "A2A0H", "NTRL0",
    "DMAP1", "BBAD1", "A1T1L", "A1T1H", "A1B1", "DAS1L", "DAS1H", "DASB1", "A2A1L", "A2A1H", "NTRL1",
    "DMAP2", "BBAD2", "A1T2L", "A1T2H", "A1B2", "DAS2L", "DAS2H", "DASB2", "A2A2L", "A2A2H", "NTRL2",
    "DMAP3", "BBAD3", "A1T3L", "A1T3H", "A1B3", "DAS3L", "DAS3H", "DASB3", "A2A3L", "A2A3H", "NTRL3",
    "DMAP4", "BBAD4", "A1T4L", "A1T4H", "A1B4", "DAS4L", "DAS4H", "DASB4", "A2A4L", "A2A4H", "NTRL4",
    "DMAP5", "BBAD5", "A1T5L", "A1T5H", "A1B5", "DAS5L", "DAS5H", "DASB5", "A2A5L", "A2A5H", "NTRL5",
    "DMAP6", "BBAD6", "A1T6L", "A1T6H", "A1B6", "DAS6L", "DAS6H", "DASB6", "A2A6L", "A2A6H", "NTRL6",
    "DMAP7", "BBAD7", "A1T7L", "A1T7H", "A1B7", "DAS7L", "DAS7H", "DASB7", "A2A7L", "A2A7H", "NTRL7",
}


def get_domain_threshold(domain: str) -> float:
    """Get appropriate quality threshold for domain."""
    return DOMAIN_THRESHOLDS.get(domain, DEFAULT_THRESHOLD)


def is_hardware_register(entity: str) -> bool:
    """Check if entity is a SNES hardware register."""
    return entity.upper() in SNES_REGISTERS


def score_code_coherence(instruction: str, output: str, domain: str) -> float:
    """Domain-aware coherence scoring.

    For code domains (asm, cpp, yaze), we don't expect word overlap.
    Instead, check if output contains code patterns.
    """
    if domain in ("asm", "cpp", "yaze", "gigaleak", "oracle"):
        # For code, just check if output looks like code
        code_indicators = [
            r'\b(lda|sta|jmp|jsr|rts|php|plp|pha|pla)\b',  # ASM mnemonics
            r'\{|\}|\(|\)',  # Braces/parens
            r';$',  # Semicolon at end (common in ASM comments)
            r'^\s*(if|for|while|return|void|int|class)',  # C++ keywords
            r'0x[0-9a-fA-F]+',  # Hex addresses
            r'\$[0-9a-fA-F]+',  # ASM hex notation
        ]

        import re
        matches = sum(1 for pattern in code_indicators if re.search(pattern, output, re.IGNORECASE | re.MULTILINE))

        # If output looks like code, give it good coherence
        if matches >= 2:
            return 0.7  # Good coherence for code
        elif matches >= 1:
            return 0.5  # Moderate
        else:
            return 0.3  # Low
    else:
        # For text domains, use word overlap (existing logic)
        instruction_words = set(instruction.lower().split())
        output_words = set(output.lower().split())

        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
        }

        instruction_words -= stopwords
        output_words -= stopwords

        if not instruction_words or not output_words:
            return 0.5

        intersection = instruction_words & output_words
        union = instruction_words | output_words

        return len(intersection) / len(union) if union else 0.5


def parse_llm_confidence(response_text: str) -> float:
    """Robustly parse confidence score from LLM response.

    Handles:
    - Plain numbers: "0.8"
    - Numbers with explanation: "0.8 - this looks accurate"
    - Numbers in sentences: "I'd rate this 0.7 confidence"
    - Invalid responses: return 0.5 (neutral)
    """
    import re

    # Try to find a decimal number between 0 and 1
    patterns = [
        r'^\s*([01]?\.\d+)\s*$',  # Plain number: "0.8"
        r'^\s*([01]?\.\d+)',  # Number at start: "0.8 because..."
        r'([01]?\.\d+)\s*$',  # Number at end: "confidence: 0.8"
        r'([01]?\.\d+)',  # Number anywhere
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text.strip())
        if match:
            try:
                value = float(match.group(1))
                if 0.0 <= value <= 1.0:
                    return value
            except ValueError:
                continue

    # Fallback: return neutral confidence
    return 0.5
