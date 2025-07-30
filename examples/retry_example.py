#!/usr/bin/env python3
"""
Example demonstrating the LLM model retry functionality.

This example shows how to configure different retry settings and how
the pipeline handles LLM model errors after max retries are exhausted.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.llm_model import LlmModel, LlmModelError


def example_basic_retry():
    """Example with default retry settings."""
    print("=== Basic Retry Example ===")

    # Create model with default retry settings (3 retries, 1s delay, 2x backoff)
    model = LlmModel.create(
        model="google/gemini-2.5-flash",
        temperature=0.2,
        max_retries=3,
        retry_delay=1.0,
        backoff_factor=2.0,
    )

    print("Model configured with:")
    print(f"  - Max retries: {model.max_retries}")
    print(f"  - Retry delay: {model.retry_delay}s")
    print(f"  - Backoff factor: {model.backoff_factor}x")
    print(f"  - Total attempts: {model.max_retries + 1}")
    print()


def example_custom_retry():
    """Example with custom retry settings."""
    print("=== Custom Retry Example ===")

    # Create model with aggressive retry settings
    model = LlmModel.create(
        model="google/gemini-2.5-flash",
        temperature=0.2,
        max_retries=5,
        retry_delay=0.5,
        backoff_factor=1.5,
    )

    print("Model configured with aggressive retry settings:")
    print(f"  - Max retries: {model.max_retries}")
    print(f"  - Retry delay: {model.retry_delay}s")
    print(f"  - Backoff factor: {model.backoff_factor}x")
    print(f"  - Total attempts: {model.max_retries + 1}")

    # Calculate delay sequence
    delays = [model.retry_delay * (model.backoff_factor**i) for i in range(model.max_retries)]
    print(f"  - Delay sequence: {delays}")
    print()


def example_no_retry():
    """Example with no retry (for testing immediate failure)."""
    print("=== No Retry Example ===")

    # Create model with no retries
    model = LlmModel.create(
        model="google/gemini-2.5-flash",
        temperature=0.2,
        max_retries=0,
        retry_delay=1.0,
        backoff_factor=2.0,
    )

    print("Model configured with no retries:")
    print(f"  - Max retries: {model.max_retries}")
    print(f"  - Total attempts: {model.max_retries + 1}")
    print()


def example_error_handling():
    """Example showing how LlmModelError is raised and handled."""
    print("=== Error Handling Example ===")

    # Create model with minimal retries for quick demonstration
    model = LlmModel.create(
        model="google/gemini-2.5-flash",
        temperature=0.2,
        max_retries=1,
        retry_delay=0.1,
        backoff_factor=1.0,
    )

    print("Attempting API call with invalid API key to demonstrate error handling...")

    try:
        # This will fail due to invalid API key
        model.chat_completion(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello, how are you?",
        )
        print("Unexpected success!")
    except LlmModelError as e:
        print(f"✅ LlmModelError caught: {e}")
        print("This error should stop the pipeline when used in a phase.")
    except Exception as e:
        print(f"❌ Unexpected error type: {type(e).__name__}: {e}")

    print()


def main():
    """Run all examples."""
    print("LLM Model Retry Functionality Examples")
    print("=" * 50)
    print()

    # Check if API key is available
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  Warning: OPENROUTER_API_KEY environment variable not set.")
        print("   Some examples may fail with authentication errors.")
        print()

    example_basic_retry()
    example_custom_retry()
    example_no_retry()
    example_error_handling()

    print("Examples completed!")
    print()
    print("Key points:")
    print("- LlmModelError is raised when max retries are exhausted")
    print("- The pipeline catches LlmModelError and stops execution")
    print("- Retry delays use exponential backoff: delay * (backoff_factor ^ attempt)")
    print("- Only network errors and 5xx/429 status codes trigger retries")


if __name__ == "__main__":
    main()
