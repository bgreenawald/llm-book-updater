#!/usr/bin/env python3
"""
Fix markdown formatting for numbered blocks in Beyond Good and Evil.

Converts numbered list items like "23. Content" into:
**23**

Content

Skips annotated sections and headers.
"""

import re
import sys
from pathlib import Path


def is_annotated_start(line: str) -> bool:
    """Check if line starts an annotated section."""
    return bool(re.match(r"^>\s*\*\*Annotated (introduction|summary):\*\*", line))


def is_annotated_end(line: str) -> bool:
    """Check if line ends an annotated section."""
    return bool(re.match(r"^>\s*\*\*End annotated (introduction|summary)\.\*\*", line))


def is_header(line: str) -> bool:
    """Check if line is a markdown header."""
    return line.startswith("#")


def fix_numbered_blocks(content: str) -> str:
    """Convert numbered items to bold headers."""
    lines = content.split("\n")
    result = []

    in_annotated = False

    for line in lines:
        # Track annotated sections - pass through unchanged
        if is_annotated_start(line):
            in_annotated = True
            result.append(line)
            continue

        if is_annotated_end(line):
            in_annotated = False
            result.append(line)
            continue

        if in_annotated:
            result.append(line)
            continue

        # Skip headers - pass through unchanged
        if is_header(line):
            result.append(line)
            continue

        # Check for numbered line: "23. content"
        match = re.match(r"^(\d+)\.\s*(.*)", line)
        if match:
            num = match.group(1)
            rest = match.group(2).strip()

            # Ensure blank line before the number (if previous line isn't already blank)
            if result and result[-1] != "":
                result.append("")

            # Convert to bold number with escaped period, inline format
            # Use \. to prevent markdown from rendering as a list
            if rest:
                result.append(f"**{num}\\.** {rest}")
            else:
                result.append(f"**{num}\\.**")
            continue

        # Default: pass through unchanged
        result.append(line)

    return "\n".join(result)


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_numbered_blocks.py <input_file> [output_file]")
        print("  If output_file is not specified, prints to stdout")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    content = input_path.read_text(encoding="utf-8")
    fixed = fix_numbered_blocks(content)

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
        output_path.write_text(fixed, encoding="utf-8")
        print(f"Fixed content written to: {output_path}")
    else:
        print(fixed)


if __name__ == "__main__":
    main()
