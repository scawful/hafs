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
        self.hex_pattern = re.compile(r"(\\$[0-9A-Fa-f]{2}:[0-9A-Fa-f]{4}|\\$[0-9A-Fa-f]{6}|\\$[0-9A-Fa-f]{4})")
        self.rep_pattern = re.compile(r"REP\s+#\$([0-9A-Fa-f]{2})", re.IGNORECASE)
        self.sep_pattern = re.compile(r"SEP\s+#\$([0-9A-Fa-f]{2})", re.IGNORECASE)
        self.label_pattern = re.compile(r"^([A-Za-z0-9_]+):")
        self.return_pattern = re.compile(r"\b(RTS|RTL|RTI)\b", re.IGNORECASE)

    def enrich(self, raw_asm: str) -> str:
        """Inject symbol names into ASM code as comments and track register widths."""
        lines = raw_asm.split('\n')
        enriched_lines = []
        
        m_8bit = True 
        x_8bit = True
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                enriched_lines.append(line)
                continue
                
            if stripped.startswith(';'):
                enriched_lines.append(line)
                continue

            if self.label_pattern.match(stripped) and not stripped.startswith('.'):
                m_8bit = True 
                x_8bit = True

            if any(stripped.lower().startswith(d) for d in ['org', 'base', 'pushpc', 'pullpc', 'check', 'architecture', 'lorom', 'hirom']):
                enriched_lines.append(line)
                continue

            # Split code and comment
            # Handle cases where the LLM might have already injected brackets incorrectly
            clean_line = line.replace('[', '; [').replace(']', ']')
            
            if ';' in clean_line:
                idx = clean_line.find(';')
                code_part = clean_line[:idx].strip()
                comment_part = clean_line[idx+1:].strip()
            else:
                code_part = clean_line.strip()
                comment_part = ""
            
            # 1. Track Width Changes
            rep_match = self.rep_pattern.search(code_part)
            sep_match = self.sep_pattern.search(code_part)
            if rep_match:
                try:
                    val = int(rep_match.group(1), 16)
                    if val & 0x20: m_8bit = False
                    if val & 0x10: x_8bit = False
                except ValueError: pass
            if sep_match:
                try:
                    val = int(sep_match.group(1), 16)
                    if val & 0x20: m_8bit = True
                    if val & 0x10: x_8bit = True
                except ValueError: pass
            
            # 2. Inject Symbols (only into comment)
            matches = self.hex_pattern.findall(code_part)
            enrichment = []
            for match in matches:
                addr = match.upper()
                name = self.symbol_map.get(addr)
                if not name and len(addr) == 5:
                    name = self.symbol_map.get(f"$7E{addr[1:]}")
                if name and name not in comment_part:
                    enrichment.append(name)
            
            # 3. Add Width Annotation
            width_anno = f"M={8 if m_8bit else 16}, X={8 if x_8bit else 16}"
            
            # Reconstruct line: <CODE> ; <WIDTH> | <SYMBOLS> | <ORIGINAL COMMENT>
            new_code = code_part
            
            anno_parts = [width_anno]
            if enrichment:
                anno_parts.append(f"Symbols: {', '.join(enrichment)}")
            if comment_part:
                anno_parts.append(comment_part)
                
            full_comment = " ; " + " | ".join(anno_parts)
            new_line = f"    {new_code.ljust(35)}{full_comment}"
            enriched_lines.append(new_line)
                
        return "\n".join(enriched_lines)
