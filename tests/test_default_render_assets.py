#!/usr/bin/env python3
"""
Test suite for default render assets functionality.

Tests that books can use default render assets when they don't define
their own, and that book-specific assets take precedence when available.
"""

import tempfile
import unittest
from pathlib import Path

from books.base_builder import BookConfig


class TestDefaultRenderAssets(unittest.TestCase):
    """Test cases for default render assets functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Ensure default assets exist for testing
        self.default_assets_dir = Path("books/default_render_assets")
        self.assertTrue(self.default_assets_dir.exists(), "Default render assets directory must exist for tests")

        # Verify all expected default assets exist
        self.default_epub_css = self.default_assets_dir / "epub.css"
        self.default_preface_md = self.default_assets_dir / "preface.md"
        self.default_license_md = self.default_assets_dir / "license.md"

        self.assertTrue(self.default_epub_css.exists(), "Default epub.css must exist")
        self.assertTrue(self.default_preface_md.exists(), "Default preface.md must exist")
        self.assertTrue(self.default_license_md.exists(), "Default license.md must exist")

    def test_default_assets_when_none_exist(self):
        """Test that default assets are used when book has no custom assets."""
        # Test with a book that doesn't have its own assets (on_liberty)
        config = BookConfig(
            name="on_liberty",
            version="1.0.0",
            title="On Liberty",
            author="John Stuart Mill",
        )

        # All assets should point to defaults
        self.assertEqual(config.epub_css, self.default_epub_css)
        self.assertEqual(config.preface_md, self.default_preface_md)
        self.assertEqual(config.license_md, self.default_license_md)

        # All assets should exist
        self.assertTrue(config.epub_css.exists())
        self.assertTrue(config.preface_md.exists())
        self.assertTrue(config.license_md.exists())

    def test_book_specific_assets_take_precedence(self):
        """Test that book-specific assets are used when they exist."""
        # Test with the_federalist_papers which has a custom preface
        config = BookConfig(
            name="the_federalist_papers",
            version="1.0.0",
            title="The Federalist Papers (Timeless Library Edition)",
            author="Alexander Hamilton, James Madison, and John Jay",
        )

        book_dir = Path("books/the_federalist_papers")

        # Check if custom preface exists (it was created in our test)
        custom_preface = book_dir / "preface.md"
        if custom_preface.exists():
            # Custom preface should be used
            self.assertEqual(config.preface_md, custom_preface)
        else:
            # Default preface should be used
            self.assertEqual(config.preface_md, self.default_preface_md)

        # Check epub.css (no custom one)
        self.assertEqual(config.epub_css, self.default_epub_css)

        # Check license.md - it might be custom or default
        custom_license = book_dir / "license.md"
        if custom_license.exists():
            # Custom license should be used
            self.assertEqual(config.license_md, custom_license)
        else:
            # Default license should be used
            self.assertEqual(config.license_md, self.default_license_md)

        # All resolved assets should exist
        self.assertTrue(config.epub_css.exists())
        self.assertTrue(config.preface_md.exists())
        self.assertTrue(config.license_md.exists())

    def test_mixed_assets_resolution(self):
        """Test asset resolution with a mix of custom and default assets."""
        # Create a temporary test book directory with some custom assets
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_book_dir = temp_path / "books" / "test_book"
            test_book_dir.mkdir(parents=True)

            # Create only a custom epub.css file
            custom_css = test_book_dir / "epub.css"
            custom_css.write_text("/* Custom CSS for test book */")

            # Temporarily change working directory for this test
            original_cwd = Path.cwd()
            try:
                import os

                os.chdir(temp_path)

                config = BookConfig(
                    name="test_book",
                    version="1.0.0",
                    title="Test Book",
                    author="Test Author",
                )

                # epub.css should use the custom one
                expected_custom_css = Path("books/test_book/epub.css")
                self.assertEqual(config.epub_css, expected_custom_css)
                self.assertTrue(config.epub_css.exists())

                # preface.md and license.md should use defaults
                expected_default_preface = Path("books/default_render_assets/preface.md")
                expected_default_license = Path("books/default_render_assets/license.md")
                self.assertEqual(config.preface_md, expected_default_preface)
                self.assertEqual(config.license_md, expected_default_license)

            finally:
                os.chdir(original_cwd)

    def test_asset_path_resolution_method(self):
        """Test the _get_asset_path method directly."""
        config = BookConfig(
            name="test_book",
            version="1.0.0",
            title="Test Book",
            author="Test Author",
        )

        # Test with a file that doesn't exist in book directory
        asset_path = config._get_asset_path("nonexistent.css")
        expected_default = Path("books/default_render_assets/nonexistent.css")
        self.assertEqual(asset_path, expected_default)

        # Test with epub.css (should exist in defaults)
        asset_path = config._get_asset_path("epub.css")
        expected_default = Path("books/default_render_assets/epub.css")
        self.assertEqual(asset_path, expected_default)


if __name__ == "__main__":
    unittest.main()
