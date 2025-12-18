# MODERNIZE Phase Optimization Plan

**Date**: December 18, 2025  
**Goal**: Enable MODERNIZE phase to work effectively with cost-efficient models through sub-block processing.

---

## Executive Summary

### Problem
MODERNIZE phase struggles with cost-effective models when processing large chapters. Models must handle extensive context windows while applying consistent modernization rules throughout.

### Solution
Split chapters into smaller paragraph-based sub-blocks for independent processing. Since modernization has no cross-paragraph dependencies, sub-blocks can be processed in parallel and concatenated.

### Expected Outcomes
- Better quality and more consistent results with cheaper models
- Reduced cost through smaller, focused context windows
- Similar or faster processing time (parallelization benefit)

---

## Core Concept

**Key Insight**: Paragraphs are natural semantic units for modernization. Group small paragraphs together or split large ones based on token counts to maintain reasonable context sizes.

Each sub-block is modernized independently, then results are concatenated. Post-processing is applied to the full reassembled chapter to ensure consistency.

This should work for both single and batch processes and work with the existing retry mechanism.

---

## Architecture Changes

### 1. Configuration

**Add to `PhaseConfig`** (`src/config.py`):
- `use_subblocks: bool = False` - Enable paragraph-based sub-block processing
- `max_subblock_tokens: int = 4096` - Maximum tokens per sub-block
- `min_subblock_tokens: int = 1024` - Minimum tokens per sub-block

**Validation**:
- Token limits must be positive and max > min
- Reasonable ranges (e.g., 1000-8000 tokens)

### 2. Processing Logic

**Add to `LlmPhase` base class** (`src/llm_phase.py`):
- `_split_block_into_subblocks()` - Entry point for splitting
- `_split_by_paragraphs()` - Core splitting algorithm

**Splitting Algorithm**:
1. Split chapter body on paragraph boundaries (`\n\n+`)
2. Count tokens for each paragraph
3. Group consecutive small paragraphs until reaching `min_subblock_tokens`
4. If single paragraph exceeds `max_subblock_tokens`, split on sentence boundaries
5. Preserve markdown structures (lists, blockquotes, code blocks)

**Modify `StandardLlmPhase._process_block()`** (`src/llm_phase.py`):
- Check if `use_subblocks` is enabled
- If yes: call new `_process_block_with_subblocks()` method
- If no: use existing single-block processing

**New method `_process_block_with_subblocks()`**:
- Split current and original bodies into sub-blocks
- Process sub-blocks in parallel (using ThreadPoolExecutor)
- Each sub-block gets its own LLM call
- Concatenate results
- Apply post-processing to full reassembled body (not individual sub-blocks)

### 3. Factory Integration

**Update `PhaseFactory.create_standard_phase()`** (`src/phase_factory.py`):
- Pass sub-block parameters from config to phase instance
- Set attributes on phase if `use_subblocks` is enabled

### 4. Usage Example

```python
PhaseConfig(
    phase_type=PhaseType.MODERNIZE,
    model=KIMI_K2,
    reasoning={"effort": "high"},
    use_subblocks=True,
    max_subblock_tokens=4096,
    min_subblock_tokens=1024,
)
```

---

## Testing Strategy

### Unit Tests
- Paragraph splitting with various configurations (small, large, mixed paragraphs)
- Paragraph grouping logic (combining small paragraphs)
- Paragraph splitting logic (splitting large paragraphs on sentence boundaries)
- Edge cases: single paragraph, very long paragraph, markdown formatting, lists
---
