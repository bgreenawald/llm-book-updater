"""Two-stage phase implementation using composition.

This module provides a TwoStageFinalPhase that implements the Phase protocol
without inheriting from LlmPhase. It uses composition with StageConfig objects
to manage the IDENTIFY and IMPLEMENT stages cleanly.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

from src.constants import DEFAULT_GENERATION_MAX_RETRIES, DEFAULT_TAGS_TO_PRESERVE
from src.cost_tracking_wrapper import add_generation_id
from src.llm_model import GenerationFailedError, LlmModel
from src.phase_utils import (
    TokenCounter,
    extract_markdown_blocks,
    get_header_and_body,
    make_llm_call_with_retry,
    read_file,
    write_file,
)
from src.post_processors import PostProcessorChain


@dataclass
class StageConfig:
    """Configuration for a single stage in a multi-stage phase.

    Attributes:
        model: LLM model instance to use for this stage
        system_prompt: Fully rendered system prompt text
        user_prompt_template: User prompt template with {placeholders}
        reasoning: Optional reasoning configuration (e.g., {"effort": "high"})
    """

    model: LlmModel
    system_prompt: str
    user_prompt_template: str
    reasoning: Optional[Dict[str, str]] = None


class TwoStageFinalPhase:
    """
    Two-stage FINAL phase using composition (not inheritance from LlmPhase).

    Implements the Phase protocol required by Pipeline through structural typing:
    - name, input_file_path, output_file_path
    - start_token_count, end_token_count
    - system_prompt_path, user_prompt_path, system_prompt
    - post_processor_chain
    - run(**kwargs)

    Internal implementation uses two stages:
    1. IDENTIFY stage: Analyzes content and produces change list (reasoning model)
    2. IMPLEMENT stage: Applies identified changes (cheaper model)
    """

    def __init__(
        self,
        name: str,
        input_file_path: Path,
        output_file_path: Path,
        original_file_path: Path,
        book_name: str,
        author_name: str,
        identify_config: StageConfig,
        implement_config: StageConfig,
        post_processor_chain: Optional[PostProcessorChain] = None,
        max_workers: Optional[int] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        use_batch: bool = False,
        batch_size: Optional[int] = None,
        enable_retry: bool = False,
        max_retries: int = DEFAULT_GENERATION_MAX_RETRIES,
        skip_if_less_than_tokens: Optional[int] = None,
        tags_to_preserve: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the two-stage FINAL phase.

        Args:
            name: Name of the phase for logging and identification
            input_file_path: Path to the input markdown file
            output_file_path: Path where processed output will be written
            original_file_path: Path to the original markdown file for reference
            book_name: Name of the book being processed
            author_name: Name of the book's author
            identify_config: Configuration for the IDENTIFY stage
            implement_config: Configuration for the IMPLEMENT stage
            post_processor_chain: Optional chain of post-processors to apply
            max_workers: Maximum number of worker threads for parallel processing
            llm_kwargs: Additional kwargs to pass to LLM calls
            use_batch: Whether to use batch processing for LLM calls
            batch_size: Number of items to process in each batch
            enable_retry: Whether to retry failed generations
            max_retries: Maximum number of retry attempts per generation
            skip_if_less_than_tokens: If set, blocks with fewer tokens will be skipped
            tags_to_preserve: Tags to preserve during processing
        """
        # Core attributes (required by Phase protocol)
        self.name = name
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.original_file_path = original_file_path

        # Token tracking (set during run())
        self.start_token_count: Optional[int] = None
        self.end_token_count: Optional[int] = None

        # For metadata collection (Protocol compatibility)
        # These are None since we handle prompts internally via StageConfig
        self.system_prompt_path: Optional[Path] = None
        self.user_prompt_path: Optional[Path] = None
        # Provide combined system prompt for metadata
        self.system_prompt = (
            f"[IDENTIFY STAGE]\n{identify_config.system_prompt}\n\n[IMPLEMENT STAGE]\n{implement_config.system_prompt}"
        )
        self.post_processor_chain = post_processor_chain

        # Stage configurations
        self.identify_config = identify_config
        self.implement_config = implement_config

        # Processing parameters
        self.book_name = book_name
        self.author_name = author_name
        self.max_workers = max_workers or 4
        self.llm_kwargs = llm_kwargs or {}
        self.use_batch = use_batch
        self.batch_size = batch_size
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.skip_if_less_than_tokens = skip_if_less_than_tokens
        self.tags_to_preserve = tags_to_preserve or list(DEFAULT_TAGS_TO_PRESERVE)

        # Utilities
        self._token_counter = TokenCounter()

        # Internal state
        self._content = ""
        self._input_text = ""
        self._original_text = ""

        # Debug data collection
        self._debug_data: List[Dict[str, Any]] = []
        self._debug_lock = threading.Lock()

        logger.info(f"TwoStageFinalPhase initialized: {name}")

    def run(self, **kwargs) -> None:
        """Execute the two-stage phase.

        Args:
            **kwargs: Additional arguments passed from Pipeline
        """
        try:
            logger.info(f"Starting two-stage phase: {self.name}")

            # Load input files
            self._input_text = read_file(self.input_file_path)
            self._original_text = read_file(self.original_file_path)

            # Count starting tokens
            self.start_token_count = self._token_counter.count(self._input_text)
            logger.info(f"Phase '{self.name}' starting with ~{self.start_token_count:,} tokens")

            # Process blocks
            self._process_markdown_blocks(**kwargs)

            # Count ending tokens
            self.end_token_count = self._token_counter.count(self._content)
            logger.info(f"Phase '{self.name}' completed with ~{self.end_token_count:,} tokens")

            # Write output
            write_file(self.output_file_path, self._content)

            # Write debug file
            self._write_debug_file()

            logger.success(f"Successfully completed two-stage phase: {self.name}")

        except Exception as e:
            logger.error(f"Failed two-stage phase {self.name}: {e}")
            raise

    def _process_markdown_blocks(self, **kwargs) -> None:
        """Process all markdown blocks through both stages."""
        current_blocks = extract_markdown_blocks(self._input_text)
        original_blocks = extract_markdown_blocks(self._original_text)

        if len(current_blocks) != len(original_blocks):
            raise ValueError(f"Block count mismatch: {len(current_blocks)} vs {len(original_blocks)}")

        if not current_blocks:
            logger.warning("No markdown blocks found")
            self._content = self._input_text
            return

        blocks = list(zip(current_blocks, original_blocks))

        # Check batch support for both models
        use_batch = (
            self.use_batch
            and hasattr(self.identify_config.model, "supports_batch")
            and self.identify_config.model.supports_batch()
            and hasattr(self.implement_config.model, "supports_batch")
            and self.implement_config.model.supports_batch()
        )

        logger.info(f"Two-stage processing: use_batch={use_batch}")

        if use_batch:
            processed = self._process_batch_mode(blocks, **kwargs)
        else:
            processed = self._process_parallel_mode(blocks, **kwargs)

        self._content = "".join(processed)

    def _process_parallel_mode(self, blocks: List[Tuple[str, str]], **kwargs) -> List[str]:
        """Process blocks in parallel (non-batch mode).

        Args:
            blocks: List of (current_block, original_block) tuples
            **kwargs: Additional context

        Returns:
            List of processed block strings
        """

        def process_one(args: Tuple[int, Tuple[str, str]]) -> str:
            idx, (current, original) = args
            return self._process_single_block(current, original, idx, **kwargs)

        indexed = list(enumerate(blocks))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_one, indexed),
                    total=len(blocks),
                    desc=f"Processing {self.name}",
                )
            )

        return results

    def _process_single_block(
        self,
        current_block: str,
        original_block: str,
        block_index: int,
        **kwargs,
    ) -> str:
        """Process one block through IDENTIFY → IMPLEMENT stages.

        Args:
            current_block: Current markdown block to process
            original_block: Original markdown block for reference
            block_index: Index of the block for tracking
            **kwargs: Additional context

        Returns:
            Processed block string
        """
        current_header, current_body = get_header_and_body(current_block)
        original_header, original_body = get_header_and_body(original_block)

        # Skip logic
        if self._should_skip(current_body, original_body, current_block):
            return f"{current_header}\n\n{current_body}\n\n"

        # Stage 1: IDENTIFY
        identify_prompt = self._format_identify_prompt(current_body, original_body, current_header)

        call_kwargs = {**self.llm_kwargs}
        if self.identify_config.reasoning:
            call_kwargs["reasoning"] = self.identify_config.reasoning

        changes_list, identify_gen_id = make_llm_call_with_retry(
            model=self.identify_config.model,
            system_prompt=self.identify_config.system_prompt,
            user_prompt=identify_prompt,
            enable_retry=self.enable_retry,
            max_retries=self.max_retries,
            block_info=f"identify: {current_header[:50]}",
            **call_kwargs,
        )

        add_generation_id(f"{self.name}_identify", identify_gen_id)

        # Store debug data
        with self._debug_lock:
            self._debug_data.append(
                {
                    "block_index": block_index,
                    "header": current_header,
                    "identify_response": changes_list,
                    "generation_id": identify_gen_id,
                }
            )

        # Stage 2: IMPLEMENT
        implement_prompt = self._format_implement_prompt(current_body, current_header, changes_list)

        processed_body, implement_gen_id = make_llm_call_with_retry(
            model=self.implement_config.model,
            system_prompt=self.implement_config.system_prompt,
            user_prompt=implement_prompt,
            enable_retry=self.enable_retry,
            max_retries=self.max_retries,
            block_info=f"implement: {current_header[:50]}",
            **self.llm_kwargs,
        )

        add_generation_id(f"{self.name}_implement", implement_gen_id)

        # Post-process
        if self.post_processor_chain:
            processed_body = self.post_processor_chain.process(
                original_block=current_body,
                llm_block=processed_body,
                **kwargs,
            )

        # Assemble result
        if processed_body.strip():
            return f"{current_header}\n\n{processed_body}\n\n"
        return f"{current_header}\n\n"

    def _should_skip(self, current_body: str, original_body: str, full_block: str) -> bool:
        """Check if block should be skipped.

        Args:
            current_body: Body of current block
            original_body: Body of original block
            full_block: Full block text for token counting

        Returns:
            True if block should be skipped
        """
        # Token threshold
        if self.skip_if_less_than_tokens:
            tokens = self._token_counter.count(full_block)
            if tokens < self.skip_if_less_than_tokens:
                logger.debug(f"Skipping block with {tokens} tokens (threshold: {self.skip_if_less_than_tokens})")
                return True

        # Empty content
        if not current_body.strip() and not original_body.strip():
            return True

        # Only special tags
        if self._is_only_tags(current_body) and self._is_only_tags(original_body):
            return True

        return False

    def _is_only_tags(self, body: str) -> bool:
        """Check if body contains only preserved tags.

        Args:
            body: Body text to check

        Returns:
            True if body contains only special tags
        """
        if not body.strip():
            return True
        lines = [line.strip() for line in body.split("\n") if line.strip()]
        return all(line in self.tags_to_preserve for line in lines) and len(lines) > 0

    def _format_identify_prompt(self, current_body: str, original_body: str, header: str) -> str:
        """Format the IDENTIFY stage user prompt.

        Args:
            current_body: Current block body
            original_body: Original block body
            header: Block header

        Returns:
            Formatted prompt string
        """
        try:
            return self.identify_config.user_prompt_template.format(
                current_body=current_body,
                original_body=original_body,
                current_header=header,
                book_name=self.book_name,
                author_name=self.author_name,
            )
        except KeyError as e:
            logger.warning(f"IDENTIFY prompt has undefined parameter: {e}")
            return self.identify_config.user_prompt_template

    def _format_implement_prompt(self, current_body: str, header: str, changes: str) -> str:
        """Format the IMPLEMENT stage user prompt.

        Args:
            current_body: Current block body
            header: Block header
            changes: Change list from IDENTIFY stage

        Returns:
            Formatted prompt string
        """
        try:
            return self.implement_config.user_prompt_template.format(
                current_body=current_body,
                current_header=header,
                changes=changes,
                book_name=self.book_name,
                author_name=self.author_name,
            )
        except KeyError as e:
            logger.warning(f"IMPLEMENT prompt has undefined parameter: {e}")
            return self.implement_config.user_prompt_template

    def _process_batch_mode(self, blocks: List[Tuple[str, str]], **kwargs) -> List[str]:
        """Process blocks using batch API calls.

        Args:
            blocks: List of (current_block, original_block) tuples
            **kwargs: Additional context

        Returns:
            List of processed block strings
        """
        if self.batch_size:
            # Process in chunks
            logger.info(f"Using batch processing with batch_size={self.batch_size}")
            processed_blocks = []
            for i in tqdm(range(0, len(blocks), self.batch_size), desc=f"Processing {self.name} (batches)"):
                chunk = blocks[i : i + self.batch_size]
                chunk_start_index = i
                chunk_results = self._process_blocks_batch(chunk, chunk_start_index, **kwargs)
                processed_blocks.extend(chunk_results)
            return processed_blocks
        else:
            # Process all at once
            logger.info("Using batch processing for all blocks at once")
            return self._process_blocks_batch(blocks, 0, **kwargs)

    def _process_blocks_batch(self, blocks: List[Tuple[str, str]], start_index: int, **kwargs) -> List[str]:
        """Process a batch of blocks using two-phase batch API calls.

        Phase 1: Batch all IDENTIFY calls → collect all change lists
        Phase 2: Batch all IMPLEMENT calls with respective change lists

        Args:
            blocks: List of (current_block, original_block) tuples
            start_index: Starting index for block numbering
            **kwargs: Additional context

        Returns:
            List of processed block strings
        """
        # Prepare block data
        block_data = []
        for i, (current_block, original_block) in enumerate(blocks):
            current_header, current_body = get_header_and_body(current_block)
            original_header, original_body = get_header_and_body(original_block)

            skip = self._should_skip(current_body, original_body, current_block)

            block_data.append(
                {
                    "index": start_index + i,
                    "current_header": current_header,
                    "current_body": current_body,
                    "original_body": original_body,
                    "skip": skip,
                }
            )

        # Phase 1: Batch IDENTIFY
        logger.info(f"Phase 1: Running IDENTIFY batch for {len(block_data)} blocks")
        identify_results = self._run_identify_batch(block_data)

        # Store debug data
        for bd, id_result in zip(block_data, identify_results):
            if not bd["skip"] and not id_result.get("skipped"):
                with self._debug_lock:
                    self._debug_data.append(
                        {
                            "block_index": bd["index"],
                            "header": bd["current_header"],
                            "identify_response": id_result.get("content", ""),
                            "generation_id": id_result.get("generation_id"),
                        }
                    )

        # Phase 2: Batch IMPLEMENT
        logger.info(f"Phase 2: Running IMPLEMENT batch for {len(block_data)} blocks")
        implement_results = self._run_implement_batch(block_data, identify_results)

        # Assemble final blocks
        processed_blocks = []
        for bd, impl_result in zip(block_data, implement_results):
            if bd["skip"] or impl_result.get("skipped"):
                processed_blocks.append(f"{bd['current_header']}\n\n{bd['current_body']}\n\n")
            else:
                processed_body = impl_result.get("content", "")

                # Apply post-processing
                if self.post_processor_chain:
                    processed_body = self.post_processor_chain.process(
                        original_block=str(bd["current_body"]),
                        llm_block=processed_body,
                        **kwargs,
                    )

                if processed_body.strip():
                    processed_blocks.append(f"{bd['current_header']}\n\n{processed_body}\n\n")
                else:
                    processed_blocks.append(f"{bd['current_header']}\n\n")

        return processed_blocks

    def _run_identify_batch(self, block_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run IDENTIFY stage as a batch API call.

        Args:
            block_data: List of block metadata dictionaries

        Returns:
            List of response dictionaries
        """
        requests: List[Optional[Dict[str, Any]]] = []
        for bd in block_data:
            if bd["skip"]:
                requests.append(None)
            else:
                user_message = self._format_identify_prompt(
                    current_body=bd["current_body"],
                    original_body=bd["original_body"],
                    header=bd["current_header"],
                )
                requests.append(
                    {
                        "system_prompt": self.identify_config.system_prompt,
                        "user_prompt": user_message,
                        "metadata": {"index": bd["index"]},
                    }
                )

        # Filter non-None requests and call batch API
        valid_requests = [r for r in requests if r is not None]
        if valid_requests:
            call_kwargs = {**self.llm_kwargs}
            if self.identify_config.reasoning:
                call_kwargs["reasoning"] = self.identify_config.reasoning

            batch_responses = self.identify_config.model.batch_chat_completion(
                requests=valid_requests,
                **call_kwargs,
            )

            # Track generation IDs
            for response in batch_responses:
                gen_id = response.get("generation_id")
                if gen_id:
                    add_generation_id(f"{self.name}_identify", gen_id)

            # Handle failed responses
            self._handle_failed_batch_responses(batch_responses, valid_requests, "identify")
        else:
            batch_responses = []

        # Create mapping from block index to response
        # This handles cases where providers return results out of order
        response_by_index: Dict[int, Dict[str, Any]] = {}
        for response in batch_responses:
            metadata = response.get("metadata", {})
            block_index = metadata.get("index")
            if block_index is not None:
                response_by_index[block_index] = response

        # Reconstruct results with placeholders for skipped blocks
        results: List[Dict[str, Any]] = []
        for req in requests:
            if req is None:
                results.append({"content": "", "skipped": True})
            else:
                block_index = req["metadata"]["index"]
                if block_index in response_by_index:
                    results.append(response_by_index[block_index])
                else:
                    # Defensive: should not happen, but handle missing response
                    logger.error(f"Missing response for block index {block_index} in IDENTIFY batch")
                    results.append({"content": "", "failed": True, "metadata": req["metadata"]})

        return results

    def _run_implement_batch(
        self, block_data: List[Dict[str, Any]], identify_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run IMPLEMENT stage as a batch API call with IDENTIFY results.

        Args:
            block_data: List of block metadata dictionaries
            identify_results: Results from IDENTIFY stage

        Returns:
            List of response dictionaries
        """
        requests: List[Optional[Dict[str, Any]]] = []
        for bd, id_result in zip(block_data, identify_results):
            if bd["skip"] or id_result.get("skipped"):
                requests.append(None)
            else:
                user_message = self._format_implement_prompt(
                    current_body=bd["current_body"],
                    header=bd["current_header"],
                    changes=id_result.get("content", ""),
                )
                requests.append(
                    {
                        "system_prompt": self.implement_config.system_prompt,
                        "user_prompt": user_message,
                        "metadata": {"index": bd["index"]},
                    }
                )

        # Filter and call batch API
        valid_requests = [r for r in requests if r is not None]
        if valid_requests:
            batch_responses = self.implement_config.model.batch_chat_completion(
                requests=valid_requests,
                **self.llm_kwargs,
            )

            # Track generation IDs
            for response in batch_responses:
                gen_id = response.get("generation_id")
                if gen_id:
                    add_generation_id(f"{self.name}_implement", gen_id)

            # Handle failed responses
            self._handle_failed_batch_responses(batch_responses, valid_requests, "implement")
        else:
            batch_responses = []

        # Create mapping from block index to response
        # This handles cases where providers return results out of order
        response_by_index: Dict[int, Dict[str, Any]] = {}
        for response in batch_responses:
            metadata = response.get("metadata", {})
            block_index = metadata.get("index")
            if block_index is not None:
                response_by_index[block_index] = response

        # Reconstruct results
        results: List[Dict[str, Any]] = []
        for req in requests:
            if req is None:
                results.append({"content": "", "skipped": True})
            else:
                block_index = req["metadata"]["index"]
                if block_index in response_by_index:
                    results.append(response_by_index[block_index])
                else:
                    # Defensive: should not happen, but handle missing response
                    logger.error(f"Missing response for block index {block_index} in IMPLEMENT batch")
                    results.append({"content": "", "failed": True, "metadata": req["metadata"]})

        return results

    def _handle_failed_batch_responses(
        self,
        batch_responses: List[Dict[str, Any]],
        valid_requests: List[Dict[str, Any]],
        stage: str,
    ) -> None:
        """Handle failed responses in batch processing.

        Args:
            batch_responses: List of batch response dictionaries
            valid_requests: List of valid request dictionaries
            stage: Stage name ("identify" or "implement")
        """
        failed_indices = [idx for idx, response in enumerate(batch_responses) if response.get("failed", False)]

        if not failed_indices:
            return

        logger.warning(
            f"{stage.upper()} batch had {len(failed_indices)} failed responses out of {len(batch_responses)}"
        )

        if not self.enable_retry:
            first_failed = batch_responses[failed_indices[0]]
            failed_content = first_failed.get("content", "Unknown error")
            raise GenerationFailedError(
                message=f"{stage.upper()} batch generation failed (retry disabled): {failed_content[:200]}",
                block_info=f"batch index {failed_indices[0]}",
            )

        # Retry failed responses individually
        logger.info(f"Retrying {len(failed_indices)} failed {stage} responses individually")
        model = self.identify_config.model if stage == "identify" else self.implement_config.model
        reasoning = self.identify_config.reasoning if stage == "identify" else None

        call_kwargs = {**self.llm_kwargs}
        if reasoning:
            call_kwargs["reasoning"] = reasoning

        for failed_idx in failed_indices:
            original_request = valid_requests[failed_idx]
            block_info = f"{stage} batch index {failed_idx}"

            retried_content, retried_gen_id = make_llm_call_with_retry(
                model=model,
                system_prompt=original_request["system_prompt"],
                user_prompt=original_request["user_prompt"],
                enable_retry=self.enable_retry,
                max_retries=self.max_retries,
                block_info=block_info,
                **call_kwargs,
            )

            # Update the response
            batch_responses[failed_idx] = {
                "content": retried_content,
                "generation_id": retried_gen_id,
                "metadata": original_request.get("metadata"),
                "failed": False,
            }

    def _write_debug_file(self) -> None:
        """Write IDENTIFY stage responses to a debug JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = self.output_file_path.parent / f"final_identify_debug_{timestamp}.json"

        # Sort debug data by block index
        sorted_data = sorted(self._debug_data, key=lambda x: x.get("block_index", 0))

        output = {
            "timestamp": datetime.now().isoformat(),
            "phase_name": self.name,
            "book_name": self.book_name,
            "author_name": self.author_name,
            "identify_model": str(self.identify_config.model),
            "implement_model": str(self.implement_config.model),
            "blocks": sorted_data,
        }

        try:
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            logger.info(f"IDENTIFY debug output written to: {debug_file}")
        except Exception as e:
            logger.error(f"Failed to write IDENTIFY debug file: {e}")
