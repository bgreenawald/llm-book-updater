# The Wealth of Nations

## How to generate

1. Download the text from: https://www.gutenberg.org/ebooks/3300 and save to `input_raw.md`
2. Remove the frontmatter and license PER the license.
3. Remove the table of contents and everything else before the Introduction
4. Run `transform.py` to get `input_transformed.py`. Run the following steps on it.
5. Turn the introduction into a Markdown 1 header
6. Add in the preface and license sections.