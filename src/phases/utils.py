"""Shared utilities for phase implementations.

This module provides common functionality used by both LlmPhase and
TwoStageFinalPhase, including token counting, file I/O, retry logic,
and markdown block processing.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
from loguru import logger

from src.llm.model import (
    GenerationFailedError,
    LlmModel,
    MaxRetriesExceededError,
    ResponseTruncatedError,
    is_failed_response,
)


class TokenCounter:
    """Utility for counting tokens in text using tiktoken."""

    def __init__(self, encoding: str = "cl100k_base"):
        """Initialize token counter with specified encoding.

        Args:
            encoding: The tiktoken encoding to use (default: cl100k_base for GPT-4)
        """
        self._tokenizer = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        """Count approximate number of tokens in text.

        Args:
            text: The text to count tokens for

        Returns:
            The approximate number of tokens
        """
        try:
            return len(self._tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting error: {e}, using char/4 fallback")
            # Fallback: rough approximation of 1 token per 4 characters
            return len(text) // 4


def should_skip_by_token_count(
    block: str,
    threshold: Optional[int],
    counter: TokenCounter,
) -> bool:
    """Check if block should be skipped based on token count.

    Args:
        block: The block to check
        threshold: Minimum token count (None to skip check)
        counter: Token counter instance

    Returns:
        True if block should be skipped
    """
    if threshold is None:
        return False

    token_count = counter.count(block)
    should_skip = token_count < threshold

    if should_skip:
        logger.debug(f"Skipping block with {token_count} tokens (threshold: {threshold})")

    return should_skip


def contains_only_special_tags(body: str, tags_to_preserve: List[str]) -> bool:
    """Check if body contains only special tags after removing blank lines.

    Args:
        body: Body content to check
        tags_to_preserve: List of special tags (e.g., ["{preface}", "{license}"])

    Returns:
        True if body contains only special tags
    """
    if not body.strip():
        return True

    lines = [line.strip() for line in body.split("\n") if line.strip()]

    for line in lines:
        if line not in tags_to_preserve:
            return False

    return len(lines) > 0


def should_skip_block(
    current_body: str,
    original_body: str,
    full_block: str,
    token_threshold: Optional[int],
    token_counter: TokenCounter,
    tags_to_preserve: List[str],
) -> bool:
    """Comprehensive check if a block should be skipped.

    Args:
        current_body: Current block body
        original_body: Original block body
        full_block: Full block text (for token counting)
        token_threshold: Minimum token count (None to skip check)
        token_counter: Token counter instance
        tags_to_preserve: List of special tags to check

    Returns:
        True if block should be skipped
    """
    if should_skip_by_token_count(full_block, token_threshold, token_counter):
        return True

    if not current_body.strip() and not original_body.strip():
        return True

    if contains_only_special_tags(current_body, tags_to_preserve) and contains_only_special_tags(
        original_body, tags_to_preserve
    ):
        return True

    return False


def read_file(path: Path) -> str:
    """Read a file with proper error handling and logging.

    Args:
        path: Path to the file to read

    Returns:
        The file contents as a string

    Raises:
        FileNotFoundError: If the file does not exist
        Exception: If there's an error reading the file
    """
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with path.open(mode="r", encoding="utf-8") as f:
            content = f.read()
        logger.debug(f"Successfully read file: {path}")
        return content
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        raise


def write_file(path: Path, content: str) -> None:
    """Write content to a file, creating parent directories if needed.

    Args:
        path: Path to write to
        content: Content to write

    Raises:
        Exception: If there's an error writing the file
    """
    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode="w", encoding="utf-8") as f:
            f.write(content)
        logger.debug(f"Successfully wrote file: {path} ({len(content)} chars)")
    except Exception as e:
        logger.error(f"Error writing file {path}: {e}")
        raise


def make_llm_call_with_retry(
    model: LlmModel,
    system_prompt: str,
    user_prompt: str,
    enable_retry: bool,
    max_retries: int,
    block_info: Optional[str] = None,
    **kwargs,
) -> Tuple[str, str]:
    """Make an LLM call with optional retry logic.

    This function handles retries for failed LLM calls according to the
    configuration. If enable_retry is False, any failure immediately raises
    an error. If True, failures are retried up to max_retries times.

    Args:
        model: The LLM model instance to use
        system_prompt: System prompt for the call
        user_prompt: User prompt for the call
        enable_retry: Whether to retry on failure
        max_retries: Maximum retry attempts (only used if enable_retry=True)
        block_info: Optional context string for error messages (e.g., "header: Introduction")
        **kwargs: Additional arguments for the LLM call (reasoning, llm_kwargs, etc.)

    Returns:
        Tuple of (response_content, generation_id)

    Raises:
        GenerationFailedError: If retry is disabled and the call fails
        MaxRetriesExceededError: If all retry attempts are exhausted
    """
    max_attempts = (max_retries + 1) if enable_retry else 1
    last_error: Optional[Exception] = None
    last_content: str = ""

    for attempt in range(max_attempts):
        try:
            content, generation_id = model.chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                **kwargs,
            )

            # Check if the response indicates a failure
            if is_failed_response(content):
                last_content = content
                error_msg = f"LLM returned failed response: {content[:100]}..."
                if enable_retry and attempt < max_retries:
                    logger.warning(
                        f"Generation failed (attempt {attempt + 1}/{max_attempts}): {error_msg}. Retrying..."
                    )
                    continue
                else:
                    # No retry or retries exhausted
                    if enable_retry:
                        raise MaxRetriesExceededError(
                            message=f"Generation failed after {max_attempts} attempts: {error_msg}",
                            attempts=max_attempts,
                            block_info=block_info,
                        )
                    else:
                        raise GenerationFailedError(
                            message=error_msg,
                            block_info=block_info,
                        )

            # Success!
            return content, generation_id

        except (GenerationFailedError, MaxRetriesExceededError):
            # Re-raise our custom exceptions (these are terminal failures)
            raise
        except ResponseTruncatedError as e:
            # Truncation is retryable - model may use fewer reasoning tokens on retry
            last_error = e
            error_msg = f"Response truncated: {e}"
            if enable_retry and attempt < max_retries:
                logger.warning(f"Response truncated (attempt {attempt + 1}/{max_attempts}): {error_msg}. Retrying...")
                continue
            else:
                # No retry or retries exhausted
                if enable_retry:
                    raise MaxRetriesExceededError(
                        message=f"Response truncated after {max_attempts} attempts: {error_msg}",
                        attempts=max_attempts,
                        block_info=block_info,
                    ) from e
                else:
                    raise GenerationFailedError(
                        message=error_msg,
                        block_info=block_info,
                    ) from e
        except Exception as e:
            last_error = e
            error_msg = f"LLM call failed: {e}"
            if enable_retry and attempt < max_retries:
                logger.warning(f"Generation failed (attempt {attempt + 1}/{max_attempts}): {error_msg}. Retrying...")
                continue
            else:
                # No retry or retries exhausted
                if enable_retry:
                    raise MaxRetriesExceededError(
                        message=f"Generation failed after {max_attempts} attempts: {error_msg}",
                        attempts=max_attempts,
                        block_info=block_info,
                    ) from e
                else:
                    raise GenerationFailedError(
                        message=error_msg,
                        block_info=block_info,
                    ) from e

    # Fallback (should never reach here, but provide safe handling)
    if last_error:
        raise GenerationFailedError(
            message=f"LLM call failed: {last_error}",
            block_info=block_info,
        ) from last_error
    else:
        raise GenerationFailedError(
            message=f"LLM returned failed response: {last_content[:100]}...",
            block_info=block_info,
        )


def extract_markdown_blocks(text: str) -> List[str]:
    """Extract markdown blocks (headers + content) from text.

    This function identifies markdown headers (# through ######) and extracts
    each header with its associated content as a separate block. It also
    preserves any preamble content that appears before the first header.

    Args:
        text: The markdown text to process

    Returns:
        List of extracted blocks, each containing a header and its content
    """
    pattern = r"(?:^|\n)(#{1,6}\s+.*?)(?=\n#{1,6}\s+|\n*$)"
    matches = list(re.finditer(pattern, text, flags=re.DOTALL))

    blocks = []
    if not matches:
        # No headers found - return entire text as single block if not empty
        if text.strip():
            blocks.append(text)
        return blocks

    # Check for preamble (content before first header)
    first_match = matches[0]
    if first_match.start() > 0:
        preamble = text[: first_match.start()]
        if preamble.strip():
            blocks.append(preamble)

    # Add matched blocks
    for match in matches:
        blocks.append(match.group(1))

    return blocks


def get_header_and_body(block: str) -> Tuple[str, str]:
    """Split a markdown block into header and body components.

    This method separates the first line (header) from the rest of the
    content (body) in a markdown block.

    Args:
        block: The markdown block to split

    Returns:
        Tuple containing (header, body)
    """
    lines = block.strip().split("\n", 1)
    header = lines[0].strip()
    body = lines[1].strip() if len(lines) > 1 else ""
    return header, body


def map_batch_responses_to_requests(
    batch_responses: List[Dict[str, Any]],
    requests: List[Optional[Dict[str, Any]]],
    stage_name: str,
    index_key: str = "block_index",
) -> List[Dict[str, Any]]:
    """Map batch responses back to original request order.

    Handles cases where providers return results out of order by creating
    a mapping from block index to response, then reconstructing results
    in the original request order.

    Args:
        batch_responses: List of response dictionaries from batch API
        requests: List of request dictionaries (None for skipped blocks)
        stage_name: Name of the stage (for error logging)
        index_key: Key to use for indexing ("block_index" or "index")

    Returns:
        List of response dictionaries in same order as requests
    """
    response_by_index: Dict[int, Dict[str, Any]] = {}
    for response in batch_responses:
        metadata = response.get("metadata", {})
        block_index = metadata.get(index_key)
        if block_index is not None:
            response_by_index[int(block_index)] = response

    results: List[Dict[str, Any]] = []
    for req in requests:
        if req is None:
            results.append({"content": "", "skipped": True})
        else:
            block_index = req["metadata"].get(index_key)
            if block_index is not None and int(block_index) in response_by_index:
                results.append(response_by_index[int(block_index)])
            else:
                logger.error(f"Missing response for {index_key} {block_index} in {stage_name.upper()} batch")
                results.append({"content": "", "failed": True, "metadata": req.get("metadata", {})})

    return results
