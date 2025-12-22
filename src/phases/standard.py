"""Standard LLM phase implementation for processing markdown content."""

from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

from loguru import logger

from src.llm.cost_tracking import add_generation_id
from src.phases.base import LlmPhase
from src.phases.utils import contains_only_special_tags, should_skip_by_token_count


class StandardLlmPhase(LlmPhase):
    """
    Standard LLM phase for processing markdown content.

    This phase processes markdown blocks by sending them to an LLM model
    and applying post-processing to clean up the results.
    """

    def _assemble_processed_block(
        self, current_header: str, current_body: str, llm_response: str, original_body: str, **kwargs
    ) -> str:
        """
        Assemble the block by replacing the body with the LLM response.

        Args:
            current_header: The header of the current block
            current_body: The body of the current block (unused in standard phase)
            llm_response: The processed content from the LLM
            original_body: The original body for reference (unused in standard phase)
            **kwargs: Additional context

        Returns:
            The block with the LLM response as the body
        """
        # Standard behavior: replace the body with the LLM response
        if llm_response.strip():
            return f"{current_header}\n\n{llm_response}\n\n"
        else:
            return f"{current_header}\n\n"

    def _process_subblock(
        self,
        current_subblock: str,
        original_body: str,
        current_header: str,
        original_header: str,
        subblock_index: int,
        **kwargs,
    ) -> str:
        """
        Process a single sub-block using the LLM model.

        Args:
            current_subblock: The current sub-block text to process
            original_body: The full original body text for reference
            current_header: The header of the block (for context in prompts)
            original_header: The original header (for context in prompts)
            subblock_index: Index of the sub-block for logging
            **kwargs: Additional context or parameters

        Returns:
            str: The processed sub-block content
        """
        # Format the user message using sub-block content
        user_message = self._format_user_message(
            current_body=current_subblock,
            original_body=original_body,
            current_header=current_header,
            original_header=original_header,
        )

        # Get LLM response with retry logic
        call_kwargs = {**self.llm_kwargs, "reasoning": self.reasoning}
        block_info = f"header: {current_header[:50]}, subblock: {subblock_index}"
        processed_subblock, generation_id = self._make_llm_call_with_retry(
            system_prompt=self.system_prompt,
            user_prompt=user_message,
            block_info=block_info,
            **call_kwargs,
        )

        # Track generation ID for cost calculation
        add_generation_id(phase_name=self.name, generation_id=generation_id)

        return processed_subblock

    def _process_block_with_subblocks(self, current_block: str, original_block: str, **kwargs) -> str:
        """
        Process a markdown block by splitting it into sub-blocks and processing each.

        This method:
        1. Extracts the header and body
        2. Splits the body into paragraphs
        3. Groups paragraphs into token-bounded sub-blocks
        4. Processes each sub-block with the LLM (in parallel if max_workers > 1)
        5. Concatenates results and applies post-processing to the full body
        6. Reassembles the block with header

        Args:
            current_block (str): The current markdown block to process
            original_block (str): The original markdown block for reference
            **kwargs: Additional context or parameters

        Returns:
            str: The processed block content
        """
        # Extract header and body from both blocks
        current_header, current_body = self._get_header_and_body(block=current_block)
        original_header, original_body = self._get_header_and_body(block=original_block)
        tags_to_preserve = self._get_tags_to_preserve()

        # Check if there's any content to process (not empty and not just special tags)
        if (not current_body.strip() and not original_body.strip()) or (
            contains_only_special_tags(current_body, tags_to_preserve)
            and contains_only_special_tags(original_body, tags_to_preserve)
        ):
            logger.debug("Empty block content or content with only special tags, returning block as-is")
            return f"{current_header}\n\n{current_body}\n\n"

        # Split current body into paragraphs
        current_paragraphs = self._split_body_into_paragraphs(current_body)

        # If no paragraphs, return as-is
        if not current_paragraphs:
            logger.debug("No paragraphs found in body, returning block as-is")
            return f"{current_header}\n\n{current_body}\n\n"

        # Group paragraphs into sub-blocks
        current_subblocks = self._group_paragraphs_into_subblocks(current_paragraphs)

        with self._subblock_stats_lock:
            self._subblock_blocks_processed_total += 1
            self._subblocks_processed_total += len(current_subblocks)
            if len(current_subblocks) > self._max_subblocks_in_single_block:
                self._max_subblocks_in_single_block = len(current_subblocks)

        logger.debug(f"Split block '{current_header[:30]}...' into {len(current_subblocks)} sub-blocks")

        # Process sub-blocks
        if self.max_workers > 1 and len(current_subblocks) > 1:
            # Parallel processing
            logger.debug(f"Processing {len(current_subblocks)} sub-blocks in parallel with {self.max_workers} workers")

            def process_subblock_wrapper(args: Tuple[int, str]) -> str:
                idx, curr_sb = args
                return self._process_subblock(
                    current_subblock=curr_sb,
                    original_body=original_body,
                    current_header=current_header,
                    original_header=original_header,
                    subblock_index=idx,
                    **kwargs,
                )

            subblock_args = [(i, curr_sb) for i, curr_sb in enumerate(current_subblocks)]

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                processed_subblocks = list(executor.map(process_subblock_wrapper, subblock_args))
        else:
            # Sequential processing
            processed_subblocks = []
            for i, curr_sb in enumerate(current_subblocks):
                processed_sb = self._process_subblock(
                    current_subblock=curr_sb,
                    original_body=original_body,
                    current_header=current_header,
                    original_header=original_header,
                    subblock_index=i,
                    **kwargs,
                )
                processed_subblocks.append(processed_sb)

        # Concatenate sub-block results with single newlines (matching input format)
        processed_body = "\n".join(processed_subblocks)

        # Apply post-processing to the full reassembled body
        processed_body = self._apply_post_processing(original_block=current_body, llm_block=processed_body, **kwargs)

        # Reconstruct the block
        if processed_body.strip():
            return f"{current_header}\n\n{processed_body}\n\n"
        logger.debug("Empty block body after sub-block processing, returning header only")
        return f"{current_header}\n\n"

    def _process_block(self, current_block: str, original_block: str, **kwargs) -> str:
        """
        Process a single markdown block using the LLM model.

        This method processes a markdown block by:
        1. Extracting the header and body
        2. Formatting a user message using the prompt template
        3. Sending the message to the LLM model (with retry if enabled)
        4. Applying post-processing to clean up the result

        If use_subblocks is enabled, delegates to _process_block_with_subblocks instead.

        Args:
            current_block (str): The current markdown block to process
            original_block (str): The original markdown block for reference
            **kwargs: Additional context or parameters

        Returns:
            str: The processed block content

        Raises:
            GenerationFailedError: If generation fails and retry is disabled
            MaxRetriesExceededError: If all retry attempts are exhausted
            EmptySectionError: If post-processing detects an invalid empty section
        """
        # Check if block should be skipped based on token count (before subblock processing)
        if should_skip_by_token_count(current_block, self.skip_if_less_than_tokens, self._token_counter):
            # Extract header and body to return block in proper format
            current_header, current_body = self._get_header_and_body(block=current_block)
            return f"{current_header}\n\n{current_body}\n\n"

        # Check if sub-block processing is enabled
        if self.use_subblocks:
            return self._process_block_with_subblocks(current_block, original_block, **kwargs)

        # Extract header and body from both blocks
        current_header, current_body = self._get_header_and_body(block=current_block)
        original_header, original_body = self._get_header_and_body(block=original_block)
        tags_to_preserve = self._get_tags_to_preserve()

        # Check if there's any content to process (not empty and not just special tags)
        if (not current_body.strip() and not original_body.strip()) or (
            contains_only_special_tags(current_body, tags_to_preserve)
            and contains_only_special_tags(original_body, tags_to_preserve)
        ):
            logger.debug("Empty block content or content with only special tags, returning block as-is")
            return f"{current_header}\n\n{current_body}\n\n"

        # Format the user message
        body = self._format_user_message(
            current_body=current_body,
            original_body=original_body,
            current_header=current_header,
            original_header=original_header,
        )

        # Get LLM response with retry logic
        # Merge reasoning and llm_kwargs
        call_kwargs = {**self.llm_kwargs, "reasoning": self.reasoning}
        block_info = f"header: {current_header[:50]}"
        processed_body, generation_id = self._make_llm_call_with_retry(
            system_prompt=self.system_prompt,
            user_prompt=body,
            block_info=block_info,
            **call_kwargs,
        )

        # Track generation ID for cost calculation
        add_generation_id(phase_name=self.name, generation_id=generation_id)

        # Apply post-processing
        processed_body = self._apply_post_processing(original_block=current_body, llm_block=processed_body, **kwargs)

        # Reconstruct the block
        if processed_body.strip():
            return f"{current_header}\n\n{processed_body}\n\n"
        logger.debug("Empty block body, returning header only")
        return f"{current_header}\n\n"
