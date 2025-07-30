#!/usr/bin/env python3


def transform_text():
    # Read the input file
    try:
        with open("input_raw.md", "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("Error: input_raw.md not found")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # Step 1: Remove whitespace at start and end of each line
    lines = [line.strip() for line in lines]

    # Step 2: Remove existing markdown headers and convert to bold
    for i, line in enumerate(lines):
        if line.startswith("#"):
            # Remove the # characters and any following spaces
            header_text = line.lstrip("#").strip()
            if header_text:  # Only convert if there's text after the #
                lines[i] = f"**{header_text}**"
            else:
                lines[i] = ""  # Empty header becomes empty line

    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Handle INTRODUCTION AND PLAN OF THE WORK special case
        if line.endswith("INTRODUCTION AND PLAN OF THE WORK."):
            result_lines.append("# INTRODUCTION AND PLAN OF THE WORK.")
            i += 1
            continue

        # Handle BOOK lines
        if line.startswith("BOOK"):
            # Capture this line and subsequent lines until next newline
            book_parts = [line]
            i += 1
            while i < len(lines) and lines[i] != "":
                book_parts.append(lines[i])
                i += 1
            # Create markdown h1 header
            book_header = "# " + " ".join(book_parts)
            result_lines.append(book_header)
            continue

        # Handle CHAPTER lines
        if line.startswith("CHAPTER"):
            # Capture this line and subsequent lines until next newline
            chapter_parts = [line]
            i += 1
            while i < len(lines) and lines[i] != "":
                chapter_parts.append(lines[i])
                i += 1
            # Create markdown h2 header
            chapter_header = "## " + " ".join(chapter_parts)
            result_lines.append(chapter_header)
            continue

        # Regular line processing
        result_lines.append(line)
        i += 1

    # Step 4: Handle newline replacement
    # Preserve blank lines, collapse multiple blank lines to single blank lines
    # Concatenate consecutive non-blank lines with spaces
    final_lines = []
    i = 0

    while i < len(result_lines):
        line = result_lines[i]

        if line == "":  # Empty line
            # Count consecutive empty lines
            empty_count = 0
            j = i
            while j < len(result_lines) and result_lines[j] == "":
                empty_count += 1
                j += 1

            # Multiple blank lines become a single blank line
            final_lines.append("")
            i = j
        else:
            # Collect consecutive non-blank lines
            non_blank_lines = []
            while i < len(result_lines) and result_lines[i] != "":
                non_blank_lines.append(result_lines[i])
                i += 1

            # Join consecutive non-blank lines with spaces
            if non_blank_lines:
                final_lines.append(" ".join(non_blank_lines))

    # Join with newlines to create final text
    final_text = "\n".join(final_lines)

    # Write the result to output file
    with open("input_transformed.md", "w", encoding="utf-8") as f:
        f.write(final_text)


if __name__ == "__main__":
    transform_text()
    print("Transformation complete. Output written to input_transformed.md")
