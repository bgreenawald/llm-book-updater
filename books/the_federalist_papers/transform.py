#!/usr/bin/env python3
"""
transform.py

Usage:
    python transform.py input.txt output.txt
"""

import argparse
import sys


def join_paragraphs(lines: list[str]) -> list[str]:
    """
    Collapse consecutive non-blank, non-header lines into a single line.
    Blank lines and lines starting with '# ' are always emitted as-is.
    """
    out: list[str] = []
    paragraph_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        # blank line: flush pending paragraph, then emit blank
        if not stripped:
            if paragraph_lines:
                combined = " ".join(line.strip() for line in paragraph_lines) + "\n"
                out.append(combined)
                paragraph_lines = []
            out.append(line)
        # markdown header: flush paragraph, then emit header
        elif line.startswith("# "):
            if paragraph_lines:
                combined = " ".join(line.strip() for line in paragraph_lines) + "\n"
                out.append(combined)
                paragraph_lines = []
            out.append(line)
        else:
            paragraph_lines.append(line)

    # flush any trailing paragraph
    if paragraph_lines:
        combined = " ".join(line.strip() for line in paragraph_lines) + "\n"
        out.append(combined)

    return out


def transform(lines: list[str]) -> list[str]:
    """
    For each "No." line:
      1. Skip the following blank(s).
      2. Collect all non-blank lines as the subject until the next blank.
      3. If the subject == "The Same Subject Continued",
         skip blank(s) again and collect until the next blank as an
         extension. Append that to the subject.
      4. Emit "## No.… - <subject>" and skip all consumed lines.
    """
    out: list[str] = []
    n = len(lines)
    i = 0

    while i < n:
        line = lines[i]
        if line.startswith("No."):
            # drop the previously emitted line
            if out:
                out.pop()

            # 1) skip blank lines after "No."
            pos = i + 1
            while pos < n and lines[pos].strip() == "":
                pos += 1

            # 2) collect subject lines until the next blank
            start = pos
            while pos < n and lines[pos].strip() != "":
                pos += 1
            end = pos  # lines[start:end] are subject lines

            subject_parts = [line.strip() for line in lines[start:end] if line.strip()]
            subject = " ".join(subject_parts)

            # skip the blank that ended the subject (if any)
            skip_to = end
            if skip_to < n and lines[skip_to].strip() == "":
                skip_to += 1

            # 3) handle continued‐subject extension
            if subject == "The Same Subject Continued":
                # skip blank(s) before extension
                ext_start = skip_to
                while ext_start < n and lines[ext_start].strip() == "":
                    ext_start += 1
                # collect extension until next blank
                ext_end = ext_start
                while ext_end < n and lines[ext_end].strip() != "":
                    ext_end += 1

                ext_parts = [line.strip() for line in lines[ext_start:ext_end] if line.strip()]
                if ext_parts:
                    subject_parts.extend(ext_parts)
                    subject = " ".join(subject_parts)

                skip_to = ext_end
                if skip_to < n and lines[skip_to].strip() == "":
                    skip_to += 1

            # 4) emit header and advance
            new_line = f"## {line.strip()} - {subject}\n\n"
            out.append(new_line)
            i = skip_to

        else:
            out.append(line)
            i += 1

    return join_paragraphs(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Collapse 'No.' lines into Markdown H2 headers")
    p.add_argument("input_file", help="Path to input .txt")
    p.add_argument("output_file", help="Path to write transformed output")
    args = p.parse_args()

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            src_lines = f.readlines()
    except OSError as e:
        sys.exit(f"Error reading {args.input_file}: {e}")

    dst_lines = transform(src_lines)

    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.writelines(dst_lines)
    except OSError as e:
        sys.exit(f"Error writing {args.output_file}: {e}")


if __name__ == "__main__":
    main()
