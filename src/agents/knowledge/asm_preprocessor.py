"""ASM Preprocessor for Embedding Enrichment.

Injects semantic meaning into raw 65816 assembly before embedding.
Includes register width tracking for M/X flags and subroutine boundary detection.
"""

import re
from typing import Dict, Optional, List, Tuple

class AsmPreprocessor:
    def __init__(self, symbol_map: Optional[Dict[str, str]] = None):
        """
        Args:
            symbol_map: Dict mapping hex addresses (e.g. "$7EF3CA") to Names ("Link_X").
        """
        self.symbol_map = symbol_map or {}
        
        # Regex for 24-bit and 16-bit hex addresses
        # Matches $XX:XXXX, $XXXXXX or $XXXX
        self.hex_pattern = re.compile(r"(\$\n[0-9A-Fa-f]{2}:[0-9A-Fa-f]{4}|\$\n[0-9A-Fa-f]{6}|\$\n[0-9A-Fa-f]{4})")
        
        # Patterns for width tracking
        self.rep_pattern = re.compile(r"REP\s+#\$([0-9A-Fa-f]{2})", re.IGNORECASE)
        self.sep_pattern = re.compile(r"SEP\s+#\$([0-9A-Fa-f]{2})", re.IGNORECASE)
        
        # Boundary markers
        self.label_pattern = re.compile(r"^([A-Za-z0-9_]+):")
        self.return_pattern = re.compile(r"\b(RTS|RTL|RTI)\b", re.IGNORECASE)

    def enrich(self, raw_asm: str) -> str:
        """Inject symbol names into ASM code inline and track register widths."""
        lines = raw_asm.split('\n')
        enriched_lines = []
        
        # State for width tracking
        m_8bit = True # Default to 8-bit for safety in ALTTP
        x_8bit = True
        
        for line in lines:
            # Skip empty lines or pure comments
            stripped = line.strip()
            if not stripped:
                enriched_lines.append(line)
                continue
                
            if stripped.startswith(';'):
                enriched_lines.append(line)
                continue

            # Detect Subroutine End: often widths are reset or irrelevant after return
            if self.return_pattern.search(stripped):
                # We don't reset state here as the routine might return in a specific state,
                # but we'll mark it in the output eventually.
                pass

            # Detect Label (Subroutine Entry): 
            # In training data, we often want to reset to a "Neutral" state at labels
            # unless it's a local label (+, -, .local)
            if self.label_pattern.match(stripped) and not stripped.startswith('.'):
                m_8bit = True # Reset to safe default at major labels
                x_8bit = True

            # Skip header directives
            if any(stripped.lower().startswith(d) for d in ['org', 'base', 'pushpc', 'pullpc', 'check', 'architecture', 'lorom', 'hirom']):
                enriched_lines.append(line)
                continue

            # Split code and comment
            if ';' in line:
                idx = line.find(';')
                code_part = line[:idx]
                comment_part = line[idx+1:]
            else:
                code_part = line
                comment_part = ""
            
            # 1. Track Width Changes
            rep_match = self.rep_pattern.search(code_part)
            sep_match = self.sep_pattern.search(code_part)
            
            if rep_match:
                try:
                    val = int(rep_match.group(1), 16)
                    if val & 0x20: m_8bit = False
                    if val & 0x10: x_8bit = False
                except ValueError:
                    pass
            
            if sep_match:
                try:
                    val = int(sep_match.group(1), 16)
                    if val & 0x20: m_8bit = True
                    if val & 0x10: x_8bit = True
                except ValueError:
                    pass
            
            # 2. Inject Symbols
            matches = self.hex_pattern.findall(code_part)
            enrichment = ""
            for match in matches:
                addr = match.upper()
                # Try absolute then relative/direct page
                name = self.symbol_map.get(addr)
                if not name and len(addr) == 5: # $XXXX
                    # Try direct page/WRAM bank 7E match
                    name = self.symbol_map.get(f"$7E{addr[1:]}")
                
                if name:
                    # Avoid duplicate label if already in comment
                    if name not in comment_part:
                        enrichment += f" [{name}]"
            
            # 3. Add Width Annotation
            width_anno = f"M={8 if m_8bit else 16}, X={8 if x_8bit else 16}"
            
            # Reconstruct line
            new_code = code_part.rstrip()
            if enrichment:
                new_code += enrichment
            
            anno_part = f"; {width_anno}"
            if comment_part.strip():
                full_comment = f"{anno_part} | {comment_part.strip()}"
            else:
                full_comment = anno_part
                
            # Align comments for readability
            new_line = f"{new_code.ljust(45)} {full_comment}"
            enriched_lines.append(new_line)
                
        return "\n".join(enriched_lines)