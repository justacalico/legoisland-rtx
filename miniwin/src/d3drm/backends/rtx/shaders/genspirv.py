#!/usr/bin/env python3
"""Embeds a compiled SPIR-V binary into a C header as a byte array."""

import argparse
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--variable", required=True)
    args = parser.parse_args()

    data = args.input.read_bytes()
    if len(data) % 4 != 0:
        raise SystemExit(f"{args.input}: SPIR-V size {len(data)} is not a multiple of 4")

    lines = [
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        f"static const uint32_t {args.variable}[] = {{",
    ]
    for i in range(0, len(data), 16):
        words = []
        for j in range(i, min(i + 16, len(data)), 4):
            words.append(f"0x{int.from_bytes(data[j:j+4], 'little'):08x}")
        lines.append("\t" + ", ".join(words) + ",")
    lines.append("};")
    lines.append(f"static const uint32_t {args.variable}_size = {len(data)};")
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
