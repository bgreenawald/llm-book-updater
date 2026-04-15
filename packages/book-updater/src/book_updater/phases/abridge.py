"""Abridged-edition phase implementations.

This module provides two phases that together produce an abridged version of a book:

1. AbridgePlanPhase — reads the full modernized book and produces a detailed
   outline: argument traced step by step, verbatim quotes, structural decisions.
   The outline is rich enough that the write phase never needs to consult the
   original source.

2. AbridgeWritePhase — reads the outline and expands each section into finished
   prose, passing already-written sections as context for voice consistency.
   Sections are processed sequentially so each section receives the full text
   of all preceding sections.

Both classes implement the Phase protocol via structural typing (no inheritance
from LlmPhase), mirroring the TwoStageFinalPhase pattern.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_core import LlmModel
from llm_core.config import DEFAULT_GENERATION_MAX_RETRIES
from loguru import logger

from book_updater.phases.utils import (
    TokenCounter,
    make_llm_call_with_retry,
    read_file,
    write_file,
)
from book_updater.processing.post_processors import PostProcessorChain

# Maximum tokens to send in a single planning call.  Books exceeding this are
# split at top-level (#) header boundaries and planned in chunks before a
# final consolidation pass.
DEFAULT_MAX_PLAN_INPUT_TOKENS = 120_000

# Regex that matches a section entry in the plan output produced by ABRIDGE_PLAN.
# Matches lines like:  ## Section 1: Some Title
_SECTION_HEADER_RE = re.compile(r"^##\s+Section\s+\d+", re.MULTILINE)


def _split_at_h1(text: str) -> List[str]:
    """Split markdown text at top-level (H1) headers.

    The preamble before the first H1 is attached to the first chunk.
    """
    h1_re = re.compile(r"^# ", re.MULTILINE)
    positions = [m.start() for m in h1_re.finditer(text)]

    if not positions:
        return [text]

    chunks: List[str] = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        chunks.append(text[start:end])

    # Attach any preamble (text before first H1) to the first chunk
    if positions[0] > 0:
        chunks[0] = text[: positions[0]] + chunks[0]

    return chunks


class AbridgePlanPhase:
    """Stage 1 of the abridged pipeline.

    Reads the entire modernized book and asks the LLM to produce a detailed
    abridgement outline: argument steps, verbatim quotes, structural decisions.
    If the book is too long for a single context window it is chunked at H1
    boundaries, each chunk is planned independently, then all chunk plans are
    consolidated into a single unified outline.

    Implements the Phase protocol (structural typing).
    """

    def __init__(
        self,
        name: str,
        input_file_path: Path,
        output_file_path: Path,
        original_file_path: Path,
        book_name: str,
        author_name: str,
        model: LlmModel,
        system_prompt_path: Optional[Path] = None,
        user_prompt_path: Optional[Path] = None,
        post_processor_chain: Optional[PostProcessorChain] = None,
        reasoning: Optional[Dict[str, str]] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        enable_retry: bool = False,
        max_retries: int = DEFAULT_GENERATION_MAX_RETRIES,
        max_input_tokens: int = DEFAULT_MAX_PLAN_INPUT_TOKENS,
    ) -> None:
        self.name = name
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.original_file_path = original_file_path
        self.book_name = book_name
        self.author_name = author_name
        self.model = model
        self.system_prompt_path = system_prompt_path or Path("./prompts/abridge_plan_system.md")
        self.user_prompt_path = user_prompt_path or Path("./prompts/abridge_plan_user.md")
        self.post_processor_chain = post_processor_chain
        self.reasoning = reasoning or {}
        self.llm_kwargs: Dict[str, Any] = llm_kwargs or {}
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.max_input_tokens = max_input_tokens

        # Phase protocol fields
        self.start_token_count: Optional[int] = None
        self.end_token_count: Optional[int] = None
        self.system_prompt: str = ""

    # ------------------------------------------------------------------
    # Phase protocol entry point
    # ------------------------------------------------------------------

    def run(self, **kwargs) -> None:
        """Execute the planning phase."""
        logger.info(f"[{self.name}] Starting abridgement planning for '{self.book_name}'")

        system_prompt = read_file(self.system_prompt_path)
        self.system_prompt = system_prompt
        user_template = read_file(self.user_prompt_path)

        full_text = read_file(self.input_file_path)

        counter = TokenCounter()
        self.start_token_count = counter.count(full_text)
        logger.info(f"[{self.name}] Input token count: {self.start_token_count:,}")

        if self.start_token_count <= self.max_input_tokens:
            plan = self._plan_single(system_prompt, user_template, full_text)
        else:
            plan = self._plan_chunked(system_prompt, user_template, full_text, counter)

        write_file(self.output_file_path, plan)
        self.end_token_count = counter.count(plan)
        logger.success(f"[{self.name}] Outline written to {self.output_file_path} ({self.end_token_count:,} tokens)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_prompt(self, template: str, full_text: str) -> str:
        return template.format(
            book_name=self.book_name,
            author_name=self.author_name,
            full_text=full_text,
        )

    def _plan_single(self, system_prompt: str, user_template: str, full_text: str) -> str:
        """Send the full book to the LLM in a single planning call."""
        user_prompt = self._build_user_prompt(user_template, full_text)
        kwargs = dict(self.llm_kwargs)
        if self.reasoning:
            kwargs["reasoning"] = self.reasoning

        content, _ = make_llm_call_with_retry(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            enable_retry=self.enable_retry,
            max_retries=self.max_retries,
            block_info="full book",
            **kwargs,
        )
        return content

    def _plan_chunked(
        self,
        system_prompt: str,
        user_template: str,
        full_text: str,
        counter: TokenCounter,
    ) -> str:
        """Plan in chunks when the book exceeds the context window.

        Splits at H1 (# ) boundaries, plans each chunk independently, then
        sends all chunk-plans to the LLM for final consolidation.
        """
        chunks = _split_at_h1(full_text)
        logger.info(f"[{self.name}] Book split into {len(chunks)} chunks for planning")

        chunk_plans: List[str] = []
        for i, chunk in enumerate(chunks):
            logger.info(f"[{self.name}] Planning chunk {i + 1}/{len(chunks)}")
            user_prompt = self._build_user_prompt(user_template, chunk)
            kwargs = dict(self.llm_kwargs)
            if self.reasoning:
                kwargs["reasoning"] = self.reasoning

            content, _ = make_llm_call_with_retry(
                model=self.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                enable_retry=self.enable_retry,
                max_retries=self.max_retries,
                block_info=f"chunk {i + 1}/{len(chunks)}",
                **kwargs,
            )
            chunk_plans.append(f"<!-- Chunk {i + 1} plan -->\n\n{content}")

        # Consolidation pass
        logger.info(f"[{self.name}] Consolidating {len(chunk_plans)} chunk plans")
        consolidation_text = "\n\n---\n\n".join(chunk_plans)
        consolidation_prompt = (
            f"Below are abridgement outlines produced for individual chunks of "
            f"*{self.book_name}* by *{self.author_name}*. "
            "Merge them into a single coherent abridgement outline, renumbering sections "
            "sequentially, eliminating duplicates, and ensuring logical flow across the "
            "full book.\n\n"
            f"{consolidation_text}"
        )
        kwargs = dict(self.llm_kwargs)
        if self.reasoning:
            kwargs["reasoning"] = self.reasoning

        final_plan, _ = make_llm_call_with_retry(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=consolidation_prompt,
            enable_retry=self.enable_retry,
            max_retries=self.max_retries,
            block_info="consolidation",
            **kwargs,
        )
        return final_plan


class AbridgeWritePhase:
    """Stage 2 of the abridged pipeline.

    Reads the outline produced by AbridgePlanPhase and expands each section
    into finished prose.  Sections are processed sequentially; each section
    receives the full text of all previously written sections so the writer
    can match established tone and avoid repetition.

    Implements the Phase protocol (structural typing).
    """

    def __init__(
        self,
        name: str,
        input_file_path: Path,
        output_file_path: Path,
        model: LlmModel,
        original_file_path: Optional[Path] = None,  # unused; kept for API compatibility
        book_name: str = "",
        author_name: str = "",
        system_prompt_path: Optional[Path] = None,
        user_prompt_path: Optional[Path] = None,
        post_processor_chain: Optional[PostProcessorChain] = None,
        reasoning: Optional[Dict[str, str]] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        max_workers: Optional[int] = None,  # unused; kept for API compatibility
        enable_retry: bool = False,
        max_retries: int = DEFAULT_GENERATION_MAX_RETRIES,
    ) -> None:
        self.name = name
        self.input_file_path = input_file_path  # outline file
        self.output_file_path = output_file_path
        self.book_name = book_name
        self.author_name = author_name
        self.model = model
        self.system_prompt_path = system_prompt_path or Path("./prompts/abridge_write_system.md")
        self.user_prompt_path = user_prompt_path or Path("./prompts/abridge_write_user.md")
        self.post_processor_chain = post_processor_chain
        self.reasoning = reasoning or {}
        self.llm_kwargs: Dict[str, Any] = llm_kwargs or {}
        self.enable_retry = enable_retry
        self.max_retries = max_retries

        # Phase protocol fields
        self.start_token_count: Optional[int] = None
        self.end_token_count: Optional[int] = None
        self.system_prompt: str = ""

    # ------------------------------------------------------------------
    # Phase protocol entry point
    # ------------------------------------------------------------------

    def run(self, **kwargs) -> None:
        """Execute the writing phase."""
        logger.info(f"[{self.name}] Starting abridgement writing for '{self.book_name}'")

        system_prompt = read_file(self.system_prompt_path)
        self.system_prompt = system_prompt
        user_template = read_file(self.user_prompt_path)

        outline_text = read_file(self.input_file_path)

        counter = TokenCounter()
        self.start_token_count = counter.count(outline_text)

        section_plans = self._parse_plan(outline_text)
        logger.info(f"[{self.name}] Outline contains {len(section_plans)} sections to write")

        written_sections = self._write_sections(
            system_prompt=system_prompt,
            user_template=user_template,
            section_plans=section_plans,
        )

        output = self._assemble_output(written_sections)
        write_file(self.output_file_path, output)
        self.end_token_count = counter.count(output)
        logger.success(
            f"[{self.name}] Abridged manuscript written to {self.output_file_path} ({self.end_token_count:,} tokens)"
        )

    # ------------------------------------------------------------------
    # Plan parsing
    # ------------------------------------------------------------------

    def _parse_plan(self, plan_text: str) -> List[Dict[str, str]]:
        """Parse the outline into a list of section dictionaries.

        Each section starts at a line matching ``## Section N``.
        Returns a list of dicts with keys ``title`` and ``body``.
        """
        lines = plan_text.splitlines(keepends=True)
        sections: List[Dict[str, str]] = []
        current_title: Optional[str] = None
        current_lines: List[str] = []

        for line in lines:
            if _SECTION_HEADER_RE.match(line):
                if current_title is not None:
                    sections.append({"title": current_title, "body": "".join(current_lines).strip()})
                current_title = line.strip()
                current_lines = []
            else:
                if current_title is not None:
                    current_lines.append(line)

        if current_title is not None and current_lines:
            sections.append({"title": current_title, "body": "".join(current_lines).strip()})

        if not sections:
            logger.warning(f"[{self.name}] Could not parse outline into sections — treating as single section")
            sections = [{"title": "## Section 1: Abridged Edition", "body": plan_text.strip()}]

        return sections

    # ------------------------------------------------------------------
    # Section writing (sequential — each section receives completed prose)
    # ------------------------------------------------------------------

    def _write_sections(
        self,
        system_prompt: str,
        user_template: str,
        section_plans: List[Dict[str, str]],
    ) -> List[str]:
        """Write all sections sequentially, passing completed prose as context."""
        written: List[str] = []

        for i, section in enumerate(section_plans):
            logger.info(f"[{self.name}] Writing section {i + 1}/{len(section_plans)}: {section['title']}")

            if written:
                completed_sections = "\n\n".join(written)
            else:
                completed_sections = "None — this is the first section."

            user_prompt = user_template.format(
                book_name=self.book_name,
                author_name=self.author_name,
                section_plan=f"{section['title']}\n\n{section['body']}",
                completed_sections=completed_sections,
            )
            kwargs = dict(self.llm_kwargs)
            if self.reasoning:
                kwargs["reasoning"] = self.reasoning

            content, _ = make_llm_call_with_retry(
                model=self.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                enable_retry=self.enable_retry,
                max_retries=self.max_retries,
                block_info=section["title"],
                **kwargs,
            )
            written.append(content)

        return written

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble_output(self, written_sections: List[str]) -> str:
        """Join written sections into the final manuscript."""
        return "\n\n".join(written_sections)
