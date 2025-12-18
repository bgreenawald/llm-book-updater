# FINAL Phase Optimization Plan

**Date**: December 18, 2025  
**Goal**: Enable FINAL phase to work effectively with cost-efficient models through two-stage decomposition.

---

## Executive Summary

### Problem
FINAL phase requires complex literary analysis and editorial decision-making. Current single-phase approach asks models to simultaneously:
- Compare Original and Transformed passages
- Identify opportunities for refinement across multiple criteria
- Make judicious edits in clear, modern prose
- Maintain voice, length constraints, and special formatting

This cognitive complexity overwhelms cheaper models.

### Solution
Split FINAL into two sequential phases:
1. **FINAL_IDENTIFY**: Use reasoning model to analyze and identify opportunities for refinement
2. **FINAL_IMPLEMENT**: Use cheaper model to mechanically apply identified changes

### Expected Outcomes
- Better quality and more consistent results
- Reduced cost through cognitive complexity decomposition
- Human-inspectable intermediate output (list of proposed changes)

---

## Core Concept

**Key Insight**: Separate complex literary analysis (expensive) from straightforward application (cheap).

**IDENTIFY stage** focuses on analytical thinking:
- Compare two versions
- Spot opportunities for improvement
- Generate structured list of proposed changes

**IMPLEMENT stage** focuses on mechanical execution:
- Read list of changes
- Apply each change to the text
- Maintain consistency and quality

---

## Architecture Changes

### 1. Configuration

**Add to `PhaseType` enum** (`src/config.py`):
- `FINAL_IDENTIFY` - Analysis stage
- `FINAL_IMPLEMENT` - Implementation stage

Keep `FINAL` for backward compatibility.

### 2. Prompt Files

Create four new prompt files in `prompts/` directory.

#### FINAL_IDENTIFY Prompts

**`prompts/final_identify_system.md`**:
- Instruct model to act as expert literary analyst
- Compare Original vs Transformed passages
- Identify specific opportunities for refinement
- Output structured list with: location, type, current text, proposed change, rationale
- Change types: Deeper Fidelity, Richer Voice, Superior Flow, Metadata Restoration, Additional Quote

**`prompts/final_identify_user.md`**:
- Provide book name, author, section header
- Include both Original and Transformed passages
- Request structured list of changes

#### FINAL_IMPLEMENT Prompts

**`prompts/final_implement_system.md`**:
- Instruct model to apply pre-identified changes
- Emphasize maintaining consistency and quality
- Preserve structure, f-string tags, quotes
- Output only the final polished passage

**`prompts/final_implement_user.md`**:
- Provide book name, author, section header
- Include changes from IDENTIFY phase
- Include Transformed passage
- Request final polished output

### 3. Phase Class

**New `FinalImplementPhase` class** (`src/llm_phase.py`):
- Extends `StandardLlmPhase`
- Constructor accepts `identify_output_file: Path` parameter
- On `run()`, loads and parses IDENTIFY output
- Stores changes in dict: `{header: changes_text}`
- Overrides `_format_user_message()` to inject relevant changes for each section

**Parsing Strategy**:
- IDENTIFY output contains changes grouped by section header (markdown `## ` headers)
- Parse into dict mapping headers to their change text
- Handle gracefully if file missing or parsing fails (log warning, continue without changes)

### 4. Factory Integration

**Update `PhaseFactory`** (`src/phase_factory.py`):

**Add to `DEFAULT_POST_PROCESSORS`**:
- `FINAL_IDENTIFY`: Minimal post-processing (validation, whitespace, preserve tags)
- `FINAL_IMPLEMENT`: Full post-processing (same as current FINAL)

**New method `create_final_implement_phase()`**:
- Similar to `create_standard_phase()` but accepts `identify_output_file` parameter
- Passes this to `FinalImplementPhase` constructor

### 5. Pipeline Integration

**Update `Pipeline._initialize_phase()`** (`src/pipeline.py`):
- Handle `FINAL_IDENTIFY` as a standard phase
- For `FINAL_IMPLEMENT`:
  - Search backwards through phases to find preceding `FINAL_IDENTIFY`
  - Get its output file path using `_get_phase_output_path()`
  - Pass to factory's `create_final_implement_phase()`
  - Raise error if no IDENTIFY phase found

### 6. Usage Example

```python
# Replace single FINAL phase with two-stage approach
PhaseConfig(
    phase_type=PhaseType.FINAL_IDENTIFY,
    model=DEEPSEEK_V32,  # Reasoning model for analysis
    reasoning={"effort": "high"},
    enable_retry=True,
),
PhaseConfig(
    phase_type=PhaseType.FINAL_IMPLEMENT,
    model=GEMINI_3_FLASH,  # Cheaper model for application
    enable_retry=True,
),
```

---

## Testing Strategy

### Unit Tests
- IDENTIFY output parsing with various formats
- Change injection into user prompts
- Error handling (missing file, malformed output)
- Dict mapping of headers to changes

### Integration Tests
- Run IDENTIFY phase on test chapter, inspect output
- Run IMPLEMENT phase with IDENTIFY output, verify changes applied
- Test full two-stage sequence
- Compare output quality vs single FINAL phase

### Prompt Iteration
- Test IDENTIFY output structure across different models
- Ensure output is parseable and actionable
- Verify IMPLEMENT can understand and apply changes
- Iterate on prompt wording based on model behavior

### Cost Analysis
- Measure cost of IDENTIFY (reasoning model, analytical task)
- Measure cost of IMPLEMENT (cheaper model, mechanical task)
- Compare total two-stage cost vs single FINAL cost
- Track quality improvements

---

## Implementation Timeline

**Week 1**:
- Days 1-2: Add new phase types and create prompt files
- Days 3-4: Implement `FinalImplementPhase` class
- Day 5: Update `PhaseFactory` and `Pipeline` integration
- Days 6-7: Initial testing and prompt iteration

**Week 2** (if needed):
- Testing with real books
- Prompt refinement based on results
- Cost and quality analysis

---

## Success Criteria

- ✅ IDENTIFY produces well-structured, actionable change lists
- ✅ IMPLEMENT successfully parses and applies changes
- ✅ Quality equivalent or better than single FINAL phase
- ✅ Measurable cost reduction (or similar cost with better quality)
- ✅ Intermediate output is human-inspectable and useful for debugging

---

## Risk Mitigation

### Risk: IDENTIFY output format inconsistent across models
**Mitigation**: Strong prompt engineering with clear format specification; robust parsing with fallbacks; consider structured output mode (JSON) if available

### Risk: IMPLEMENT can't parse IDENTIFY output
**Mitigation**: Graceful error handling; log warnings but don't fail; fallback to processing without changes if parsing fails completely

### Risk: Two-stage more expensive than single-stage
**Mitigation**: Benchmark costs before full rollout; if savings don't materialize, can revert to single FINAL; consider adjusting model choices (cheaper IMPLEMENT model)

---

## Rollout Strategy

### 1. Feature Flag
Two-stage FINAL is opt-in. Existing `PhaseType.FINAL` continues to work unchanged.

### 2. Gradual Migration
- Test on 1-2 books initially
- Manually inspect IDENTIFY output quality
- Verify IMPLEMENT applies changes correctly
- Gather quality and cost metrics
- Expand to more books if successful

### 3. Backward Compatibility
- Keep `PhaseType.FINAL` working as before
- New configs explicitly use `FINAL_IDENTIFY` + `FINAL_IMPLEMENT`
- Pipeline validates IMPLEMENT has preceding IDENTIFY

### 4. Documentation
- Update README with two-stage approach
- Add example to `books/README.md`
- Document model recommendations for each stage
- Include sample IDENTIFY output format

---

## Open Questions

1. **IDENTIFY output format**: JSON vs Markdown? (Consider model capabilities and parsing robustness)
2. **Structured output**: Should we use JSON mode for IDENTIFY? (More reliable parsing, but not all models support it)
3. **Change granularity**: Sentence-level, paragraph-level, or mixed?
4. **Error recovery**: What to do if IMPLEMENT fails on specific changes?
5. **Quality metrics**: How to systematically compare single vs two-stage quality?

---

## Next Steps

1. Review and approve this plan
2. Make any necessary adjustments to scope or approach
3. Begin implementation (start with prompts, test with sample text)
4. Create prototype IDENTIFY output to validate format
5. Implement phase class and integration
6. Iterate based on test results

