# How to Generate

1. Download raw file from https://www.gutenberg.org/ebooks/18 and store in 'input_raw.md'
2. Strip out all references to Project Gutenberg and the license PER the license.
3. Strip out table of contents and frontmatter (will be added back in later)
4. Run `transform.py` to a new file named `input_transformed.md`. This will be the input file to the LLM.
5. Add the preface and license placeholders to the front and back of the book respectively.
