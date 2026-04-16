"""Abridged-edition phase implementations.

This module provides three phases that together produce an abridged version of a book:

1. AbridgePlanPhase — reads the full modernized book and produces a detailed
   outline: argument traced step by step, verbatim quotes, structural decisions,
   and source chapter references for each section.

2. AbridgeFleshPhase — enriches each skeletal section plan into a detailed writing
   brief, using only the source chapters referenced by that section. Each section
   is processed independently and in parallel.

3. AbridgeWritePhase — reads the fleshed-out plan and expands each section into
   finished prose. First generates a writing profile from a sample of the
   modernized text (for voice consistency), then writes all sections in parallel
   using that profile as shared context.

All classes implement the Phase protocol via structural typing (no inheritance
from LlmPhase), mirroring the TwoStageFinalPhase pattern.
"""

import re
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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

# Default token limit for the writing profile sample (Stage 3a).
DEFAULT_PROFILE_TOKEN_LIMIT = 10_000

# Regex that matches a section entry in the plan output produced by ABRIDGE_PLAN.
# Matches lines like:  ## Section 1: Some Title
_SECTION_HEADER_RE = re.compile(r"^##\s+Section\s+\d+", re.MULTILINE)

# Stopwords removed when normalising chapter title strings for fuzzy matching.
_TITLE_STOPWORDS: Set[str] = {
    "chapter",
    "part",
    "section",
    "the",
    "a",
    "an",
    "of",
    "and",
    "in",
    "to",
}

# Roman numeral tokens (up to ~20) recognised as trivial and removed during normalisation.
_ROMAN_NUMERALS: Set[str] = {
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
    "xi",
    "xii",
    "xiii",
    "xiv",
    "xv",
    "xvi",
    "xvii",
    "xviii",
    "xix",
    "xx",
}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


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


def _normalize_title(s: str) -> Set[str]:
    """Return the set of significant words from a heading/chapter title.

    Lowercases, strips punctuation, removes stopwords, roman numerals, and
    pure-digit tokens.  Used for fuzzy chapter matching in Stage 2.
    """
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    tokens = s.split()
    result: Set[str] = set()
    for tok in tokens:
        if tok in _TITLE_STOPWORDS:
            continue
        if tok in _ROMAN_NUMERALS:
            continue
        if tok.isdigit():
            continue
        result.add(tok)
    return result


def _parse_source_chapters(section_body: str) -> List[str]:
    """Extract the **Source chapters:** field from a section plan body.

    Returns a list of chapter title strings, or an empty list if the field is
    absent (which will trigger the fallback path in _extract_chapters_for_section).
    """
    match = re.search(r"\*\*Source chapters:\*\*\s*(.+)", section_body)
    if not match:
        return []
    raw = match.group(1).strip()
    # Chapters are comma-separated on a single line
    chapters = [c.strip() for c in raw.split(",") if c.strip()]
    return chapters


def _extract_chapters_for_section(
    full_text: str,
    chapter_refs: List[str],
    max_tokens: int,
    counter: TokenCounter,
) -> str:
    """Extract only the source chapters relevant to a given section plan.

    Algorithm:
    1. Scan full_text for all headings and record (start_pos, depth, title).
    2. For each referenced chapter title, find the best-matching heading via
       normalised word-overlap (≥50% of non-trivial words must match).
    3. For each matched heading, extract from its position to the next heading
       at the same or shallower depth (capturing all nested sub-sections).
    4. Deduplicate and sort ranges; concatenate extracted text.
    5. Truncate to max_tokens if needed.
    6. Fall back to full text (truncated) if no chapters can be matched.
    """
    heading_re = re.compile(r"^(#{1,6})\s+(.+)", re.MULTILINE)
    headings: List[tuple] = []  # (start_pos, depth, title_text)
    for m in heading_re.finditer(full_text):
        depth = len(m.group(1))
        title_text = m.group(2).strip()
        headings.append((m.start(), depth, title_text))

    if not headings:
        # No headings at all — fall back to full text truncated
        return _truncate_to_tokens(full_text, max_tokens, counter)

    # Build a lookup of normalised heading titles
    norm_headings = [(start, depth, title, _normalize_title(title)) for start, depth, title in headings]

    matched_ranges: List[tuple] = []  # (start, end) character ranges

    for ref in chapter_refs:
        ref_words = _normalize_title(ref)
        if not ref_words:
            continue

        for idx, (start, depth, title, norm_title) in enumerate(norm_headings):
            if not norm_title:
                continue
            overlap = len(ref_words & norm_title)
            if overlap / len(ref_words) >= 0.5:
                # Found a match — determine range to next sibling/parent heading
                end_pos = len(full_text)
                for future_start, future_depth, _, _ in norm_headings[idx + 1 :]:
                    if future_depth <= depth:
                        end_pos = future_start
                        break
                matched_ranges.append((start, end_pos))
                break  # use first match per reference

    if not matched_ranges:
        return _truncate_to_tokens(full_text, max_tokens, counter)

    # Deduplicate and sort
    matched_ranges = sorted(set(matched_ranges))

    # Merge overlapping ranges
    merged: List[tuple] = [matched_ranges[0]]
    for start, end in matched_ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    extracted = "".join(full_text[s:e] for s, e in merged)
    return _truncate_to_tokens(extracted, max_tokens, counter)


def _truncate_to_tokens(text: str, max_tokens: int, counter: TokenCounter) -> str:
    """Truncate text to approximately max_tokens, appending a note if truncated."""
    if counter.count(text) <= max_tokens:
        return text
    # Approximate: 4 chars per token; refine with binary search
    approx_chars = max_tokens * 4
    truncated = text[:approx_chars]
    # Tighten until under limit (at most a few iterations)
    while counter.count(truncated) > max_tokens and len(truncated) > 100:
        approx_chars = int(approx_chars * 0.85)
        truncated = text[:approx_chars]
    return truncated + "\n\n[...source text truncated to fit context window...]"


# ---------------------------------------------------------------------------
# Stage 1 — AbridgePlanPhase
# ---------------------------------------------------------------------------


class AbridgePlanPhase:
    """Stage 1 of the abridged pipeline.

    Reads the entire modernized book and asks the LLM to produce a detailed
    abridgement outline: argument steps, verbatim quotes, source chapter
    references, and structural decisions.  If the book is too long for a
    single context window it is chunked at H1 boundaries, each chunk is
    planned independently, then all chunk plans are consolidated into a single
    unified outline.

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
        self.system_prompt_path: Path = system_prompt_path or Path("./prompts/abridge_plan_system.md")
        self.user_prompt_path: Path = user_prompt_path or Path("./prompts/abridge_plan_user.md")
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


# ---------------------------------------------------------------------------
# Stage 2 — AbridgeFleshPhase
# ---------------------------------------------------------------------------


class AbridgeFleshPhase:
    """Stage 2 of the abridged pipeline.

    Reads the skeletal section plans produced by AbridgePlanPhase and enriches
    each one into a detailed writing brief.  For each section it extracts only
    the source chapters referenced by that section from the full modernized text,
    then asks the LLM to flesh out the plan in detail.

    Sections are processed fully in parallel via ThreadPoolExecutor.

    Implements the Phase protocol (structural typing).
    """

    def __init__(
        self,
        name: str,
        input_file_path: Path,
        output_file_path: Path,
        modernized_text_path: Path,
        original_file_path: Path,
        book_name: str,
        author_name: str,
        model: LlmModel,
        system_prompt_path: Optional[Path] = None,
        user_prompt_path: Optional[Path] = None,
        post_processor_chain: Optional[PostProcessorChain] = None,
        reasoning: Optional[Dict[str, str]] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        max_workers: Optional[int] = None,
        enable_retry: bool = False,
        max_retries: int = DEFAULT_GENERATION_MAX_RETRIES,
        max_input_tokens: int = DEFAULT_MAX_PLAN_INPUT_TOKENS,
    ) -> None:
        self.name = name
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.modernized_text_path = modernized_text_path
        self.original_file_path = original_file_path
        self.book_name = book_name
        self.author_name = author_name
        self.model = model
        self.system_prompt_path: Path = system_prompt_path or Path("./prompts/abridge_flesh_system.md")
        self.user_prompt_path: Path = user_prompt_path or Path("./prompts/abridge_flesh_user.md")
        self.post_processor_chain = post_processor_chain
        self.reasoning = reasoning or {}
        self.llm_kwargs: Dict[str, Any] = llm_kwargs or {}
        self.max_workers = max_workers
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
        """Execute the flesh-out phase."""
        logger.info(f"[{self.name}] Starting abridgement flesh-out for '{self.book_name}'")

        system_prompt = read_file(self.system_prompt_path)
        self.system_prompt = system_prompt
        user_template = read_file(self.user_prompt_path)

        plan_text = read_file(self.input_file_path)
        full_text = read_file(self.modernized_text_path)

        counter = TokenCounter()
        self.start_token_count = counter.count(plan_text)
        logger.info(f"[{self.name}] Plan token count: {self.start_token_count:,}")

        section_plans = self._parse_plan(plan_text)
        logger.info(f"[{self.name}] Plan contains {len(section_plans)} sections to flesh out")

        fleshed_sections = self._flesh_sections(
            system_prompt=system_prompt,
            user_template=user_template,
            section_plans=section_plans,
            full_text=full_text,
            counter=counter,
        )

        output = "\n\n".join(fleshed_sections)
        write_file(self.output_file_path, output)
        self.end_token_count = counter.count(output)
        logger.success(
            f"[{self.name}] Fleshed plan written to {self.output_file_path} ({self.end_token_count:,} tokens)"
        )

    # ------------------------------------------------------------------
    # Plan parsing (shared with AbridgeWritePhase)
    # ------------------------------------------------------------------

    def _parse_plan(self, plan_text: str) -> List[Dict[str, str]]:
        """Parse the plan into a list of section dictionaries."""
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
            logger.warning(f"[{self.name}] Could not parse plan into sections — treating as single section")
            sections = [{"title": "## Section 1: Abridged Edition", "body": plan_text.strip()}]

        return sections

    # ------------------------------------------------------------------
    # Section fleshing (parallel)
    # ------------------------------------------------------------------

    def _flesh_one_section(
        self,
        section: Dict[str, str],
        section_index: int,
        total: int,
        system_prompt: str,
        user_template: str,
        full_text: str,
        counter: TokenCounter,
    ) -> str:
        """Flesh out a single section plan.  Called in parallel."""
        logger.info(f"[{self.name}] Fleshing section {section_index + 1}/{total}: {section['title']}")

        chapter_refs = _parse_source_chapters(section["body"])
        source_chapters = _extract_chapters_for_section(full_text, chapter_refs, self.max_input_tokens, counter)

        user_prompt = user_template.format(
            book_name=self.book_name,
            author_name=self.author_name,
            section_plan=f"{section['title']}\n\n{section['body']}",
            source_chapters=source_chapters,
        )
        call_kwargs = dict(self.llm_kwargs)
        if self.reasoning:
            call_kwargs["reasoning"] = self.reasoning

        content, _ = make_llm_call_with_retry(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            enable_retry=self.enable_retry,
            max_retries=self.max_retries,
            block_info=section["title"],
            **call_kwargs,
        )

        if self.post_processor_chain:
            content = self.post_processor_chain.process(
                original_block=section["body"],
                llm_block=content,
            )

        return f"{section['title']}\n\n{content}"

    def _flesh_sections(
        self,
        system_prompt: str,
        user_template: str,
        section_plans: List[Dict[str, str]],
        full_text: str,
        counter: TokenCounter,
    ) -> List[str]:
        """Flesh out all sections in parallel."""
        total = len(section_plans)
        results: Dict[int, str] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._flesh_one_section,
                    section,
                    i,
                    total,
                    system_prompt,
                    user_template,
                    full_text,
                    counter,
                ): i
                for i, section in enumerate(section_plans)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return [results[i] for i in range(total)]


# ---------------------------------------------------------------------------
# Stage 3 — AbridgeWritePhase
# ---------------------------------------------------------------------------


class AbridgeWritePhase:
    """Stage 3 of the abridged pipeline.

    Two internal sub-stages run sequentially within a single run() call:

    3a — Writing Profile: generates a detailed style profile from a sample of
         the full modernized text.  Runs once before section writing begins.
         The profile is saved as a side file for inspection.

    3b — Section Writing: expands each fleshed-out section plan into final
         prose using the writing profile for voice consistency.  Sections are
         processed in parallel; they no longer depend on each other.

    Implements the Phase protocol (structural typing).
    """

    def __init__(
        self,
        name: str,
        input_file_path: Path,
        output_file_path: Path,
        model: LlmModel,
        profile_model: LlmModel,
        modernized_text_path: Path,
        original_file_path: Optional[Path] = None,
        book_name: str = "",
        author_name: str = "",
        system_prompt_path: Optional[Path] = None,
        user_prompt_path: Optional[Path] = None,
        profile_system_prompt_path: Optional[Path] = None,
        profile_user_prompt_path: Optional[Path] = None,
        post_processor_chain: Optional[PostProcessorChain] = None,
        reasoning: Optional[Dict[str, str]] = None,
        profile_reasoning: Optional[Dict[str, str]] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        max_workers: Optional[int] = None,
        enable_retry: bool = False,
        max_retries: int = DEFAULT_GENERATION_MAX_RETRIES,
        profile_token_limit: int = DEFAULT_PROFILE_TOKEN_LIMIT,
    ) -> None:
        self.name = name
        self.input_file_path = input_file_path  # fleshed plan file
        self.output_file_path = output_file_path
        self.modernized_text_path = modernized_text_path
        self.book_name = book_name
        self.author_name = author_name
        self.model = model  # write sub-stage model
        self.profile_model = profile_model  # profile sub-stage model
        self.system_prompt_path: Path = system_prompt_path or Path("./prompts/abridge_write_system.md")
        self.user_prompt_path: Path = user_prompt_path or Path("./prompts/abridge_write_user.md")
        self.profile_system_prompt_path: Path = profile_system_prompt_path or Path(
            "./prompts/abridge_profile_system.md"
        )
        self.profile_user_prompt_path: Path = profile_user_prompt_path or Path("./prompts/abridge_profile_user.md")
        self.post_processor_chain = post_processor_chain
        self.reasoning = reasoning or {}
        self.profile_reasoning = profile_reasoning or {}
        self.llm_kwargs: Dict[str, Any] = llm_kwargs or {}
        self.max_workers = max_workers
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.profile_token_limit = profile_token_limit

        # Phase protocol fields
        self.start_token_count: Optional[int] = None
        self.end_token_count: Optional[int] = None
        self.system_prompt: str = ""

    # ------------------------------------------------------------------
    # Phase protocol entry point
    # ------------------------------------------------------------------

    def run(self, **kwargs) -> None:
        """Execute the writing phase (profile sub-stage then section writing)."""
        logger.info(f"[{self.name}] Starting abridgement writing for '{self.book_name}'")

        write_system_prompt = read_file(self.system_prompt_path)
        self.system_prompt = write_system_prompt
        write_user_template = read_file(self.user_prompt_path)

        fleshed_text = read_file(self.input_file_path)

        counter = TokenCounter()
        self.start_token_count = counter.count(fleshed_text)

        # Stage 3a — generate writing profile
        logger.info(f"[{self.name}] Generating writing profile")
        writing_profile = self._generate_writing_profile(counter)

        # Save profile side file for inspection
        profile_path = self.output_file_path.parent / f"{self.output_file_path.stem}_writing_profile.md"
        write_file(profile_path, writing_profile)
        logger.info(f"[{self.name}] Writing profile saved to {profile_path}")

        # Stage 3b — parse fleshed sections and write in parallel
        section_plans = self._parse_plan(fleshed_text)
        logger.info(f"[{self.name}] Fleshed plan contains {len(section_plans)} sections to write")

        written_sections = self._write_sections(
            system_prompt=write_system_prompt,
            user_template=write_user_template,
            section_plans=section_plans,
            writing_profile=writing_profile,
        )

        output = self._assemble_output(written_sections)
        write_file(self.output_file_path, output)
        self.end_token_count = counter.count(output)
        logger.success(
            f"[{self.name}] Abridged manuscript written to {self.output_file_path} ({self.end_token_count:,} tokens)"
        )

    # ------------------------------------------------------------------
    # Stage 3a — writing profile generation
    # ------------------------------------------------------------------

    def _generate_writing_profile(self, counter: TokenCounter) -> str:
        """Generate a writing style profile from a sample of the modernized text."""
        profile_system = read_file(self.profile_system_prompt_path)
        profile_user_template = read_file(self.profile_user_prompt_path)

        full_text = read_file(self.modernized_text_path)
        text_sample = _truncate_to_tokens(full_text, self.profile_token_limit, counter)

        user_prompt = profile_user_template.format(
            book_name=self.book_name,
            author_name=self.author_name,
            text_sample=text_sample,
        )
        call_kwargs = dict(self.llm_kwargs)
        if self.profile_reasoning:
            call_kwargs["reasoning"] = self.profile_reasoning

        profile, _ = make_llm_call_with_retry(
            model=self.profile_model,
            system_prompt=profile_system,
            user_prompt=user_prompt,
            enable_retry=self.enable_retry,
            max_retries=self.max_retries,
            block_info="writing profile",
            **call_kwargs,
        )
        return profile

    # ------------------------------------------------------------------
    # Plan parsing
    # ------------------------------------------------------------------

    def _parse_plan(self, plan_text: str) -> List[Dict[str, str]]:
        """Parse the fleshed outline into a list of section dictionaries."""
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
            logger.warning(f"[{self.name}] Could not parse fleshed plan into sections — treating as single section")
            sections = [{"title": "## Section 1: Abridged Edition", "body": plan_text.strip()}]

        return sections

    # ------------------------------------------------------------------
    # Stage 3b — section writing (parallel)
    # ------------------------------------------------------------------

    def _write_one_section(
        self,
        section: Dict[str, str],
        section_index: int,
        total: int,
        system_prompt: str,
        user_template: str,
        writing_profile: str,
    ) -> str:
        """Write a single section.  Called in parallel."""
        logger.info(f"[{self.name}] Writing section {section_index + 1}/{total}: {section['title']}")

        user_prompt = user_template.format(
            book_name=self.book_name,
            author_name=self.author_name,
            section_plan=f"{section['title']}\n\n{section['body']}",
            writing_profile=writing_profile,
        )
        call_kwargs = dict(self.llm_kwargs)
        if self.reasoning:
            call_kwargs["reasoning"] = self.reasoning

        content, _ = make_llm_call_with_retry(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            enable_retry=self.enable_retry,
            max_retries=self.max_retries,
            block_info=section["title"],
            **call_kwargs,
        )

        if self.post_processor_chain:
            content = self.post_processor_chain.process(
                original_block=section["body"],
                llm_block=content,
            )

        return content

    def _write_sections(
        self,
        system_prompt: str,
        user_template: str,
        section_plans: List[Dict[str, str]],
        writing_profile: str,
    ) -> List[str]:
        """Write all sections in parallel using the shared writing profile."""
        total = len(section_plans)
        results: Dict[int, str] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._write_one_section,
                    section,
                    i,
                    total,
                    system_prompt,
                    user_template,
                    writing_profile,
                ): i
                for i, section in enumerate(section_plans)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return [results[i] for i in range(total)]

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble_output(self, written_sections: List[str]) -> str:
        """Join written sections into the final manuscript."""
        return "\n\n".join(written_sections)
