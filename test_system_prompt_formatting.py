#!/usr/bin/env python3
"""
Test script to verify system prompt formatting with special f-string tags.
"""

import sys
from pathlib import Path

# Add project root to path to allow importing from src
sys.path.append(str(Path(__file__).parent))

from src.llm_phase import LlmPhase
from src.post_processors import PostProcessorChain, PreserveFStringTagsProcessor


class TestLlmPhase(LlmPhase):
    """Test implementation of LlmPhase for testing system prompt formatting."""

    def _process_block(self, current_block: str, original_block: str, **kwargs) -> str:
        """Dummy implementation for testing."""
        return current_block


def test_system_prompt_formatting():
    """Test that system prompt formatting handles special f-string tags correctly."""

    print("=== Testing System Prompt Formatting ===")

    # Create test files
    test_files = {
        "test_system_prompt.md": """
# Test System Prompt

This is a test system prompt that contains special f-string tags.

* Preserve all special f-string tags such as `{preface}`, `{license}`, and any similar tags exactly as they appear in the original text.
* The length reduction should be {length_reduction}.
* Book: {book_name} by {author_name}
* Custom tag: {dedication}
""",
        "dummy.md": "# Test\n\nTest content",
        "dummy_original.md": "# Test\n\nOriginal content",
        "dummy_user.md": "Test user prompt",
    }

    # Write test files
    for filename, content in test_files.items():
        with Path(filename).open("w", encoding="utf-8") as f:
            f.write(content)

    try:
        # Test 1: Default tags_to_preserve
        print("\n--- Test 1: Default tags_to_preserve ---")
        post_processor_chain = PostProcessorChain()
        post_processor_chain.add_processor(PreserveFStringTagsProcessor())

        phase = TestLlmPhase(
            name="test",
            input_file_path=Path("dummy.md"),
            output_file_path=Path("dummy_output.md"),
            original_file_path=Path("dummy_original.md"),
            system_prompt_path=Path("test_system_prompt.md"),
            user_prompt_path=Path("dummy_user.md"),
            book_name="Test Book",
            author_name="Test Author",
            model=None,
            length_reduction=30,
            post_processor_chain=post_processor_chain,
        )

        formatted_prompt = phase._read_system_prompt()
        print("Formatted prompt:")
        print(formatted_prompt)

        # Test 2: Custom tags_to_preserve
        print("\n--- Test 2: Custom tags_to_preserve ---")
        custom_post_processor_chain = PostProcessorChain()
        custom_post_processor_chain.add_processor(
            PreserveFStringTagsProcessor(config={"tags_to_preserve": ["{preface}", "{license}", "{dedication}"]})
        )

        phase2 = TestLlmPhase(
            name="test2",
            input_file_path=Path("dummy.md"),
            output_file_path=Path("dummy_output.md"),
            original_file_path=Path("dummy_original.md"),
            system_prompt_path=Path("test_system_prompt.md"),
            user_prompt_path=Path("dummy_user.md"),
            book_name="Test Book",
            author_name="Test Author",
            model=None,
            length_reduction=30,
            post_processor_chain=custom_post_processor_chain,
        )

        formatted_prompt2 = phase2._read_system_prompt()
        print("Formatted prompt with custom tags:")
        print(formatted_prompt2)

        # Test 3: No length_reduction
        print("\n--- Test 3: No length_reduction ---")
        phase3 = TestLlmPhase(
            name="test3",
            input_file_path=Path("dummy.md"),
            output_file_path=Path("dummy_output.md"),
            original_file_path=Path("dummy_original.md"),
            system_prompt_path=Path("test_system_prompt.md"),
            user_prompt_path=Path("dummy_user.md"),
            book_name="Test Book",
            author_name="Test Author",
            model=None,
            length_reduction=None,
            post_processor_chain=post_processor_chain,
        )

        formatted_prompt3 = phase3._read_system_prompt()
        print("Formatted prompt without length_reduction:")
        print(formatted_prompt3)

        print("\n=== All tests completed successfully! ===")

    finally:
        # Clean up test files
        for filename in test_files.keys():
            if Path(filename).exists():
                Path(filename).unlink()


if __name__ == "__main__":
    test_system_prompt_formatting()
