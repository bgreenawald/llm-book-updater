"""Utility functions for LLM Core library."""


def is_failed_response(content: str) -> bool:
    """Check if an LLM response indicates a failure.

    A response is considered failed if:
    - It is empty or contains only whitespace
    - It starts with "Error:" (batch error marker)

    Args:
        content: The response content to check

    Returns:
        bool: True if the response indicates a failure, False otherwise
    """
    if not content or not content.strip():
        return True
    if content.strip().startswith("Error:"):
        return True
    return False
