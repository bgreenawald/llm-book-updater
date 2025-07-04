# Book Build System

This directory contains a generalizable build system for creating consistent book builds across different projects. The system provides shared logic while maintaining flexibility for each unique book.

## Overview

The build system consists of:

- **`base_builder.py`** - Shared base class containing all common build logic
- **`build_template.py`** - Template showing how to create book-specific build scripts
- **`the_federalist_papers/build.py`** - Example implementation for The Federalist Papers

## Key Features

### Consistent Naming Conventions
- All builds follow the same file naming pattern: `{Clean-Title}-{type}.{extension}`
- Consistent directory structure across all books
- Standardized metadata handling

### Shared Build Logic
- Markdown formatting with preface and license insertion
- HTML tag cleaning (`<br>` replacement)
- Annotation pattern removal
- EPUB and PDF generation using Pandoc and Calibre
- Staging and cleanup processes

### Book-Specific Flexibility
- Custom source file locations
- Different original file naming conventions
- Book-specific metadata and titles
- Extensible for additional book-specific processing

## Directory Structure

```
books/
├── base_builder.py              # Shared base class
├── build_template.py            # Template for new books
├── README.md                    # This file
└── {book_name}/                 # Book-specific directory
    ├── build.py                 # Book-specific build script
    ├── output/                  # Generated output files
    ├── preface.md               # Book preface
    ├── license.md               # License information
    ├── cover.png                # Book cover image
    ├── epub.css                 # EPUB styling
    └── staging/                 # Temporary build files (auto-created)
```

## Creating a New Book Build

### 1. Create Book Directory
```bash
mkdir books/your_book_name
cd books/your_book_name
```

### 2. Copy Template
```bash
cp ../build_template.py build.py
```

### 3. Customize the Build Script

Edit `build.py` to implement the abstract methods:

```python
class YourBookBuilder(BaseBookBuilder):
    def get_source_files(self) -> Dict[str, Path]:
        """Specify where your source files are located."""
        return {
            "modernized": self.config.source_output_dir / "your-modernized-file.md",
            "annotated": self.config.source_output_dir / "your-annotated-file.md",
        }

    def get_original_file(self) -> Optional[Path]:
        """Specify where your original file is located."""
        return self.config.source_output_dir / "your-original-file.md"
```

### 4. Configure Book Metadata

Update the `build()` function with your book's information:

```python
config = BookConfig(
    name="your_book_name",
    version="1.0.0",
    title="Your Book Title",
    author="Your Book Author",
    # Optional: custom clean title for filenames
    # clean_title="Your-Custom-Clean-Title"
)
```

### 5. Add Required Assets

Ensure you have the following files in your book directory:
- `preface.md` - Book preface content
- `license.md` - License information
- `cover.png` - Book cover image
- `epub.css` - EPUB styling

## Running a Build

### Command Line Usage
```bash
# From the book directory
python build.py build 1.0.0 --name your_book_name

# Or from the root directory
python books/your_book_name/build.py build 1.0.0 --name your_book_name
```

### Build Output

The build process creates:

1. **Staging files** in `books/{book_name}/staging/`
2. **Base directory files** in `books/{book_name}/`:
   - `output-modernized.md`
   - `output-annotated.md`
   - `output-original.md` (if available)
   - `metadata.json`
3. **Final build artifacts** in `build/{book_name}/{version}/`:
   - `{Clean-Title}-modernized.md`
   - `{Clean-Title}-annotated.md`
   - `{Clean-Title}-original.md` (if available)
   - `{Clean-Title}-metadata.json`
   - `{Clean-Title}-modernized.epub`
   - `{Clean-Title}-modernized.pdf`
   - `{Clean-Title}-annotated.epub`
   - `{Clean-Title}-annotated.pdf`

## Extending the System

### Adding Custom Processing

You can extend the base builder by overriding methods in your book-specific class:

```python
class YourBookBuilder(BaseBookBuilder):
    def custom_processing(self) -> None:
        """Add custom processing steps."""
        # Your custom logic here
        pass

    def build(self) -> None:
        """Override build to add custom steps."""
        # Call parent build
        super().build()
        # Add custom processing
        self.custom_processing()
```

### Custom File Types

To support additional file types, extend the `get_source_files()` method:

```python
def get_source_files(self) -> Dict[str, Path]:
    return {
        "modernized": self.config.source_output_dir / "modernized.md",
        "annotated": self.config.source_output_dir / "annotated.md",
        "summary": self.config.source_output_dir / "summary.md",  # New type
    }
```

## Dependencies

The build system requires:
- `pypandoc` - For EPUB generation
- `loguru` - For logging
- `Calibre` - For PDF generation (must have `ebook-convert` in PATH)

## Example: The Federalist Papers

See `the_federalist_papers/build.py` for a complete example implementation that demonstrates:

- Standard source file naming conventions
- Original file detection using pattern matching
- Book-specific metadata configuration
- Integration with the shared build system

## Benefits

1. **Consistency** - All books follow the same build process and naming conventions
2. **Maintainability** - Shared logic is centralized and easy to update
3. **Flexibility** - Each book can customize source file locations and processing
4. **Reusability** - New books can be added quickly using the template
5. **Reliability** - Proven build logic is shared across all books 