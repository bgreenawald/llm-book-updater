import difflib
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from loguru import logger


class PostProcessor(ABC):
    """
    Abstract base class for post-processors that clean up LLM-generated content.

    Post-processors take the original block and LLM-generated block and apply
    various cleanup operations to fix errors, improve formatting, or ensure
    consistency.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the post-processor.

        Args:
            name (str): Name of the post-processor for logging and identification
            config (Optional[Dict[str, Any]]): Configuration parameters for the processor
        """
        self.name = name
        self.config = config or {}
        logger.debug(f"Initialized post-processor: {name}")

    @abstractmethod
    def process(self, original_block: str, llm_block: str, **kwargs) -> str:
        """
        Process the LLM-generated block using the original block as reference.

        Args:
            original_block (str): The original markdown block before LLM processing
            llm_block (str): The block generated by the LLM
            **kwargs: Additional context or parameters

        Returns:
            str: The post-processed block
        """
        pass

    def __str__(self) -> str:
        """
        Returns a string representation of the PostProcessor instance.
        """
        return f"{self.__class__.__name__}(name={self.name})"

    def __repr__(self) -> str:
        """
        Returns a detailed string representation of the PostProcessor instance for debugging.
        """
        return self.__str__()


class EnsureBlankLineProcessor(PostProcessor):
    """
    Ensures there is a blank line between any two elements, with exceptions for
    markdown lists and multiline block quotes.

    Quote and Annotation blocks are single-line blocks that should be surrounded
    by blank lines.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="ensure_blank_line", config=config)
        self.list_pattern = re.compile(r"^\s*([*\-+]|\d+\.)\s+")

    def process(self, original_block: str, llm_block: str, **kwargs) -> str:
        """
        Processes the LLM-generated block to ensure proper blank line separation.

        Args:
            original_block (str): The original markdown block (unused in this processor).
            llm_block (str): The block generated by the LLM.
            **kwargs: Additional context or parameters (unused in this processor).

        Returns:
            str: The post-processed block with correct blank line separation.
        """
        lines = llm_block.split("\n")
        processed_lines = []

        for i, line in enumerate(lines):
            processed_lines.append(line)

            if i < len(lines) - 1:
                next_line = lines[i + 1]

                if line.strip() and next_line.strip():
                    # Check for exceptions
                    is_list = self.list_pattern.match(line) and self.list_pattern.match(next_line)

                    # Check if we're in a multiline quote block (not Quote/Annotation blocks)
                    in_multiline_quote = self._is_in_multiline_quote(
                        lines=lines, current_idx=i, current_line=line, next_line=next_line
                    )

                    if not is_list and not in_multiline_quote:
                        processed_lines.append("")

        return "\n".join(processed_lines)

    def _is_in_multiline_quote(self, lines: List[str], current_idx: int, current_line: str, next_line: str) -> bool:
        """
        Determine if we're in a multiline quote block that should not have
        blank lines inserted between its lines.

        Note: Quote and Annotation blocks are single-line blocks and are
        handled separately.
        """
        # Check if current line starts with quote marker
        if not current_line.strip().startswith(">"):
            return False

        # Check if this is a Quote or Annotation block (single-line)
        if "**Quote:**" in current_line or "**Annotation:**" in current_line:
            return False

        # Check if next line also starts with quote marker
        if next_line.strip().startswith(">"):
            return True

        # Check if we're in the middle of a multiline quote block
        # Look ahead to see if there are more quote lines coming
        for j in range(current_idx + 2, len(lines)):
            future_line = lines[j].strip()
            if not future_line:
                continue
            if future_line.startswith(">"):
                # But not if it's a Quote or Annotation block
                if not ("**Quote:**" in future_line or "**Annotation:**" in future_line):
                    return True
            # If we hit a non-quote line, we're not in a multiline quote
            break

        return False


class RemoveBlankLinesInListProcessor(PostProcessor):
    """
    Removes blank lines between elements in a Markdown list (either ordered or unordered).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="remove_blank_lines_in_list", config=config)
        self.list_pattern = re.compile(r"^\s*([*\-+]|\d+\.)\s+")

    def _is_list_related(self, line: str) -> bool:
        return bool(self.list_pattern.match(line)) or (line.strip() and line.startswith("  "))

    def process(self, original_block: str, llm_block: str, **kwargs) -> str:
        lines = llm_block.split("\n")
        processed_lines = []
        for i, line in enumerate(lines):
            if not line.strip():  # if blank line
                # find previous non-blank line
                prev_line = None
                for j in range(i - 1, -1, -1):
                    if lines[j].strip():
                        prev_line = lines[j]
                        break

                # find next non-blank line
                next_line = None
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        next_line = lines[j]
                        break

                if prev_line and next_line and self._is_list_related(prev_line) and self._is_list_related(next_line):
                    continue  # skip blank line

            processed_lines.append(line)
        return "\n".join(processed_lines)


class RemoveXmlTagsProcessor(PostProcessor):
    """
    Removes XML tags from LLM-generated content, except for <br> tags.

    This processor uses a regular expression to identify and remove XML tags
    while preserving <br> and <br/> tags which are commonly used for line breaks
    in markdown content.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RemoveXmlTagsProcessor.

        Args:
            config (Optional[Dict[str, Any]]): Configuration parameters for the processor
        """
        super().__init__(name="remove_xml_tags", config=config)
        self.xml_tag_pattern = re.compile(pattern=r"<(?!(/)?br)[^>]*>")

    def process(self, original_block: str, llm_block: str, **kwargs) -> str:
        """
        Remove XML tags from the LLM-generated block.

        This method processes the LLM-generated content to remove all XML tags
        except for <br> and <br/> tags, which are preserved for markdown formatting.

        Args:
            original_block (str): The original markdown block (unused in this processor)
            llm_block (str): The block generated by the LLM containing XML tags
            **kwargs: Additional context or parameters (unused in this processor)

        Returns:
            str: The processed block with XML tags removed, preserving <br> tags
        """
        return self.xml_tag_pattern.sub(repl="", string=llm_block)


class RemoveTrailingWhitespaceProcessor(PostProcessor):
    """
    Removes trailing whitespace from each line.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="remove_trailing_whitespace", config=config)

    def process(self, original_block: str, llm_block: str, **kwargs) -> str:
        return "\n".join(line.rstrip() for line in llm_block.split("\n"))


class RevertRemovedBlockLines(PostProcessor):
    """
    Restores block comment lines (starting with '> ') that were removed by the LLM.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="revert_removed_block_lines", config=config)

    def process(self, original_block: str, llm_block: str, **kwargs) -> str:
        original_lines = original_block.splitlines()
        llm_lines = llm_block.splitlines()

        matcher = difflib.SequenceMatcher(a=original_lines, b=llm_lines)
        processed_lines = list(llm_lines)

        for tag, i1, i2, j1, j2 in reversed(matcher.get_opcodes()):
            if tag == "delete":
                for i in range(i2 - 1, i1 - 1, -1):
                    line = original_lines[i]
                    if line.strip().startswith(">"):
                        logger.info(f"Restoring deleted block line: '{line}'")
                        processed_lines.insert(j1, line)

        return "\n".join(processed_lines)


class NoNewHeadersPostProcessor(PostProcessor):
    """
    Ensures no new markdown headers are added to the content.

    This processor handles two cases:
    1. An entirely new markdown header is added: The new header line is deleted.
    2. An existing line is converted to a markdown header: The line is reverted to its original state.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="no_new_headers", config=config)
        self.header_pattern = re.compile(r"^(#+)\s+(.*)")

    def process(self, original_block: str, llm_block: str, **kwargs) -> str:
        original_lines = original_block.splitlines()
        original_lines_set = set(original_lines)
        original_content_map = {line.strip(): line for line in original_lines}

        llm_lines = llm_block.splitlines()
        processed_lines = []

        for line in llm_lines:
            match = self.header_pattern.match(line)

            if not match:
                processed_lines.append(line)
                continue

            if line in original_lines_set:
                processed_lines.append(line)
                continue

            header_content = match.group(2).strip()

            if header_content in original_content_map:
                original_line = original_content_map[header_content]
                logger.info(f"Reverting converted header: '{line}' to '{original_line}'")
                processed_lines.append(original_line)
            else:
                logger.info(f"Removing new header: '{line}'")

        return "\n".join(processed_lines)


class PostProcessorChain(PostProcessor):
    """
    A chain of post-processors that are applied in sequence.

    Each post-processor in the chain receives the output of the previous
    post-processor, allowing for complex multi-step cleanup operations.
    """

    def __init__(self, processors: Optional[List[PostProcessor]] = None):
        """
        Initialize the post-processor chain.

        Args:
            processors (Optional[List[PostProcessor]]): List of post-processors to chain
        """
        super().__init__(name="post_processor_chain")
        self.processors = processors or []
        logger.debug(f"Initialized post-processor chain with {len(self.processors)} processors")

    def add_processor(self, processor: PostProcessor) -> None:
        """
        Add a post-processor to the end of the chain.

        Args:
            processor (PostProcessor): The post-processor to add
        """
        self.processors.append(processor)
        logger.debug(f"Added post-processor '{processor.name}' to chain")

    def process(self, original_block: str, llm_block: str, **kwargs) -> str:
        """
        Apply all post-processors in the chain sequentially.

        Args:
            original_block (str): The original markdown block
            llm_block (str): The initial LLM-generated block
            **kwargs: Additional context passed to all processors

        Returns:
            str: The final post-processed block
        """
        current_block = llm_block

        logger.debug(f"Starting post-processing chain with {len(self.processors)} processors")

        for i, processor in enumerate(self.processors):
            try:
                logger.debug(f"Applying post-processor {i + 1}/{len(self.processors)}: {processor.name}")
                current_block = processor.process(original_block, current_block, **kwargs)
                logger.debug(f"Post-processor {processor.name} completed successfully")
            except Exception as e:
                logger.error(f"Error in post-processor {processor.name}: {str(e)}")
                logger.exception("Post-processor error stack trace")
                # Continue with the chain, using the block as-is
                continue

        logger.debug(f"Post-processing chain completed. Final block length: {len(current_block)} characters")
        return current_block

    def __len__(self) -> int:
        return len(self.processors)

    def __str__(self) -> str:
        """
        Returns a string representation of the PostProcessorChain instance.
        """
        return f"PostProcessorChain({[p.name for p in self.processors]})"

    def __repr__(self) -> str:
        """
        Returns a detailed string representation of the PostProcessorChain instance for debugging.
        """
        return self.__str__()


class OrderQuoteAnnotationProcessor(PostProcessor):
    """
    Reorders Quote and Annotation blocks so that all quotes come before all
    annotations within uninterrupted blocks of quotes and annotations.

    Blank lines separate uninterrupted blocks. Within each block, quotes are
    ordered first (in their original order), followed by annotations (in their
    original order).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="order_quote_annotation", config=config)

    def process(self, original_block: str, llm_block: str, **kwargs) -> str:
        """
        Reorders Quote and Annotation blocks within the LLM-generated content.

        Args:
            original_block (str): The original markdown block (unused in this processor).
            llm_block (str): The block generated by the LLM.
            **kwargs: Additional context or parameters (unused in this processor).

        Returns:
            str: The post-processed block with reordered quotes and annotations.
        """
        lines = llm_block.split("\n")
        processed_lines = []
        current_block = []
        i = 0
        n = len(lines)

        def flush_block():
            if current_block:
                processed_lines.extend(self._reorder_block(current_block))
                current_block.clear()

        while i < n:
            line = lines[i]
            if self._is_quote_or_annotation_block(line):
                current_block.append(line)
                i += 1
            elif not line.strip():  # Blank line
                if current_block:
                    # Look ahead for next non-blank line
                    j = i + 1
                    while j < n and not lines[j].strip():
                        j += 1
                    if j < n and self._is_quote_or_annotation_block(lines[j]):
                        # Next non-blank is quote/annotation, skip this blank line
                        i += 1
                        continue

                    flush_block()
                    processed_lines.append(line)
                    i += 1
                else:
                    processed_lines.append(line)
                    i += 1
            else:
                flush_block()
                processed_lines.append(line)
                i += 1
        flush_block()
        return "\n".join(processed_lines)

    def _is_quote_or_annotation_block(self, line: str) -> bool:
        """Check if a line is a Quote or Annotation block."""
        stripped = line.strip()
        return stripped.startswith("> **Quote:") or stripped.startswith("> **Annotation:")

    def _reorder_block(self, quote_annotation_lines: List[str]) -> List[str]:
        """
        Reorder Quote/Annotation lines while removing blank lines between blocks
        """
        # Separate quotes and annotations
        quotes = []
        annotations = []

        for line in quote_annotation_lines:
            if line.strip().startswith("> **Quote:"):
                quotes.append(line)
            elif line.strip().startswith("> **Annotation:"):
                annotations.append(line)

        # Create the reordered Quote/Annotation lines
        reordered_qa_lines = quotes + annotations

        return reordered_qa_lines


class PreserveFStringTagsProcessor(PostProcessor):
    """
    Preserves special Python f-string tags like {preface} and {license} that may be
    removed by the LLM during processing. These tags only appear on their own lines.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="preserve_fstring_tags", config=config)
        self.tags_to_preserve = (
            config.get("tags_to_preserve", ["{preface}", "{license}"]) if config else ["{preface}", "{license}"]
        )

    def process(self, original_block: str, llm_block: str, **kwargs) -> str:
        """
        Restores lines that are exactly a tag if they were removed by the LLM, and also restores any
        immediately adjacent blank lines that were deleted, to preserve the original block structure
        and spacing.
        Args:
            original_block (str): The original markdown block containing f-string tags
            llm_block (str): The block generated by the LLM
            **kwargs: Additional context or parameters
        Returns:
            str: The post-processed block with preserved f-string tag lines and spacing
        """
        original_lines = original_block.splitlines()
        llm_lines = llm_block.splitlines()

        # Check if there are multiple occurrences of the same tag
        original_tags = [line for line in original_lines if self._is_tag_line(line)]
        has_multiple_occurrences = len(original_tags) != len(set(original_tags))

        # Check if there's any similarity between original and LLM blocks
        matcher = difflib.SequenceMatcher(a=original_lines, b=llm_lines)
        has_similarity = any(tag == "equal" for tag, _, _, _, _ in matcher.get_opcodes())

        if has_similarity and not has_multiple_occurrences:
            # If there's similarity and no multiple occurrences, try to restore tags and adjacent
            # blank lines at their original positions
            processed_lines = list(llm_lines)
            for tag, i1, i2, j1, j2 in reversed(matcher.get_opcodes()):
                if tag == "delete":
                    # Collect lines to restore (tags and adjacent blank lines)
                    i = i2 - 1
                    while i >= i1:
                        line = original_lines[i]
                        if self._is_tag_line(line) or line.strip() == "":
                            logger.info(f"Restoring deleted line: '{line}'")
                            processed_lines.insert(j1, line)
                        i -= 1
            result = "\n".join(processed_lines)
        else:
            # If no similarity or multiple occurrences, append missing tags at the end
            result = llm_block

        # Check if any tags are still missing and append them at the end
        missing_tags = self._find_missing_tags(original_block, result)
        if missing_tags:
            for tag in missing_tags:
                logger.info(f"Appending missing f-string tag: '{tag}'")
                result += f"\n{tag}"
        return result

    def _is_tag_line(self, line: str) -> bool:
        """Check if a line is a tag line, handling whitespace variations."""
        stripped = line.strip()
        return stripped in self.tags_to_preserve

    def _find_missing_tags(self, original_block: str, processed_block: str) -> List[str]:
        """Find tags that are in the original but missing from the processed block."""
        original_lines = original_block.splitlines()
        processed_lines = processed_block.splitlines()

        original_tags = [line for line in original_lines if self._is_tag_line(line)]
        processed_tags = [line for line in processed_lines if self._is_tag_line(line)]

        # Count occurrences to handle multiple instances of the same tag
        from collections import Counter

        original_tag_counts = Counter(original_tags)
        processed_tag_counts = Counter(processed_tags)

        missing_tags = []
        for tag, count in original_tag_counts.items():
            missing_count = count - processed_tag_counts.get(tag, 0)
            missing_tags.extend([tag] * missing_count)

        return missing_tags
