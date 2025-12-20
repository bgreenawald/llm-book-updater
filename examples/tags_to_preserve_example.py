"""
Example script demonstrating how to use the tags_to_preserve configuration.

This example shows how to configure which f-string tags should be preserved
during processing, such as {preface} and {license} tags.
"""

import sys
from pathlib import Path
from typing import List

# Add project root to path to allow importing from src
sys.path.append(str(Path(__file__).parent.parent))

from src.config import PhaseConfig, PhaseType, RunConfig
from src.constants import LLM_DEFAULT_TEMPERATURE
from src.llm_model import GEMINI_PRO, OPENAI_04_MINI


def main():
    """Demonstrate different tags_to_preserve configurations."""

    # Example 1: Default tags_to_preserve (["{preface}", "{license}"])
    print("=== Example 1: Default tags_to_preserve ===")
    config1 = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=[PhaseConfig(phase_type=PhaseType.MODERNIZE)],
        # Uses default: tags_to_preserve=["{preface}", "{license}"]
    )
    print(f"Config 1 tags_to_preserve: {config1.tags_to_preserve}")
    print()

    # Example 2: Custom tags_to_preserve
    print("=== Example 2: Custom tags_to_preserve ===")
    config2 = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=[PhaseConfig(phase_type=PhaseType.MODERNIZE)],
        tags_to_preserve=["{preface}", "{license}", "{dedication}", "{acknowledgments}"],
    )
    print(f"Config 2 tags_to_preserve: {config2.tags_to_preserve}")
    print()

    # Example 3: Only specific tags
    print("=== Example 3: Only specific tags ===")
    config3 = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=[PhaseConfig(phase_type=PhaseType.MODERNIZE)],
        tags_to_preserve=["{preface}"],  # Only preserve preface tag
    )
    print(f"Config 3 tags_to_preserve: {config3.tags_to_preserve}")
    print()

    # Example 4: No tags to preserve
    print("=== Example 4: No tags to preserve ===")
    config4 = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=[PhaseConfig(phase_type=PhaseType.MODERNIZE)],
        tags_to_preserve=[],  # Don't preserve any tags
    )
    print(f"Config 4 tags_to_preserve: {config4.tags_to_preserve}")
    print()

    # Example 5: Complex pipeline with custom tags
    print("=== Example 5: Complex pipeline with custom tags ===")
    run_phases: List[PhaseConfig] = [
        PhaseConfig(
            phase_type=PhaseType.MODERNIZE,
            model=OPENAI_04_MINI,
            temperature=LLM_DEFAULT_TEMPERATURE,
        ),
        PhaseConfig(
            phase_type=PhaseType.EDIT,
            model=GEMINI_PRO,
            temperature=LLM_DEFAULT_TEMPERATURE,
        ),
        PhaseConfig(
            phase_type=PhaseType.FINAL,
            model=OPENAI_04_MINI,
            temperature=LLM_DEFAULT_TEMPERATURE,
        ),
    ]

    config5 = RunConfig(
        book_name="Complex Example Book",
        author_name="Complex Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=run_phases,
        tags_to_preserve=["{preface}", "{license}", "{dedication}"],
    )
    print(f"Config 5 tags_to_preserve: {config5.tags_to_preserve}")
    print()

    print("Benefits of tags_to_preserve configuration:")
    print("- Configurable at the run level for all phases")
    print("- Defaults to ['{preface}', '{license}'] if not specified")
    print("- Can be customized per run based on document requirements")
    print("- Preserves important structural elements during processing")
    print("- Works with all phase types (MODERNIZE, EDIT, ANNOTATE, etc.)")

    # Note: In a real scenario, you would run the pipeline with one of these configs
    # run_pipeline(config1)  # or config2, config3, etc.


if __name__ == "__main__":
    main()
