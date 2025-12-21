#!/usr/bin/env python3
"""
Convert an Asar-generated WLA symbol file into a Mesen .mlb label file.

Typical use (LoROM, headerless):
    python Tools/build_mlb_from_symbols.py Roms/oos91x.symbols --output Roms/labels/oos91x.mlb

Options let you change mapping (lorom/hirom), add a header bias, or keep low-bank symbols.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple


Symbol = Tuple[int, int, str]  # bank, address, name


def parse_symbols(path: Path) -> Iterator[Symbol]:
    """Yield (bank, address, name) from a WLA .symbols file."""
    pattern = re.compile(r"([0-9A-Fa-f]{2}):([0-9A-Fa-f]{4})\s+:([\w.@$]+)")
    for line in path.read_text().splitlines():
        m = pattern.match(line)
        if not m:
            continue
        yield int(m.group(1), 16), int(m.group(2), 16), m.group(3)


def to_pc(
    bank: int,
    addr: int,
    mapping: str,
    header_bytes: int,
    include_low: bool,
) -> Optional[int]:
    """
    Convert SNES bank:addr to PC offset.

    - LoROM: banks are 0x8000-byte halves; by default we skip <0x8000 mirrors unless include_low is true.
    - HiROM: full 0x10000-byte banks.
    """
    if mapping == "lorom":
        if addr < 0x8000 and not include_low:
            return None
        if addr < 0x8000:
            pc = bank * 0x10000 + addr  # mirror access; rarely wanted
        else:
            pc = bank * 0x8000 + (addr - 0x8000)
    else:  # hirom
        pc = bank * 0x10000 + addr
    return pc + header_bytes


def build_mlb(
    symbols: Iterable[Symbol],
    mapping: str,
    header_bytes: int,
    include_low: bool,
    prefix: str,
) -> str:
    lines = []
    for bank, addr, name in symbols:
        pc = to_pc(bank, addr, mapping, header_bytes, include_low)
        if pc is None:
            continue
        lines.append(f"{prefix}:{pc:06X}:{name}")
    return "\n".join(lines) + ("\n" if lines else "")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", type=Path, help="Input WLA .symbols file from Asar")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output .mlb path for Mesen",
    )
    parser.add_argument(
        "--mapping",
        choices=["lorom", "hirom"],
        default="lorom",
        help="ROM mapping used by the project",
    )
    parser.add_argument(
        "--header-bytes",
        type=int,
        default=0,
        help="Add this many bytes to PC offsets if your ROM has a header (e.g., 512)",
    )
    parser.add_argument(
        "--include-low",
        action="store_true",
        help="Include <$8000 symbols for LoROM (mirrors/vectors); skipped by default",
    )
    parser.add_argument(
        "--prefix",
        default="SnesPrgRom",
        help="Segment prefix written to the .mlb lines",
    )
    args = parser.parse_args(argv)

    if not args.symbols.exists():
        parser.error(f"symbols file not found: {args.symbols}")

    mlb_text = build_mlb(
        symbols=parse_symbols(args.symbols),
        mapping=args.mapping,
        header_bytes=args.header_bytes,
        include_low=args.include_low,
        prefix=args.prefix,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(mlb_text)
    print(f"Wrote {args.output} ({len(mlb_text.splitlines())} labels)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
