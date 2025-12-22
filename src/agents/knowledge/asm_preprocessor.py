"""ASM Preprocessor for Embedding Enrichment.

Injects semantic meaning into raw 65816 assembly before embedding.
"""

import re
from typing import Dict, Optional

class AsmPreprocessor:
    def __init__(self, symbol_map: Optional[Dict[str, str]] = None):
        """
        Args:
            symbol_map: Dict mapping hex addresses (e.g. "$7EF3CA") to Names ("Link_X").
        """
        self.symbol_map = symbol_map or {}
        
        # Regex for 24-bit and 16-bit hex addresses
        # Matches $XX:XXXX, $XXXXXX or $XXXX
        self.hex_pattern = re.compile(r"(\$[0-9A-Fa-f]{2}:[0-9A-Fa-f]{4}|\$[0-9A-Fa-f]{6}|\$[0-9A-Fa-f]{4})")

    def enrich(self, raw_asm: str) -> str:
        """Inject symbol names into ASM code inline."""
        lines = raw_asm.split('\n')
        enriched_lines = []
        
        for line in lines:
            # Skip comments
            code_part = line.split(';')[0]
            comment_part = line.split(';')[1] if ';' in line else ""
            
            # Find all hex addresses
            matches = self.hex_pattern.findall(code_part)
            
            enrichment = ""
            for match in matches:
                # Normalize address (naive)
                addr = match.upper()
                if addr in self.symbol_map:
                    enrichment += f" [{self.symbol_map[addr]}]"
            
            # Reconstruct line
            if enrichment:
                # Add enrichment before the comment, or at the end of code
                new_line = f"{code_part.rstrip()}{enrichment}"
                if comment_part:
                    new_line += f" ; {comment_part}"
                enriched_lines.append(new_line)
            else:
                enriched_lines.append(line)
                
        return "\n".join(enriched_lines)

    def expand_opcodes(self, raw_asm: str) -> str:
        """Optional: Expand cryptic opcodes to text."""
        # TODO: Implement mapping for SEP/REP/LDA/STA
        return raw_asm
