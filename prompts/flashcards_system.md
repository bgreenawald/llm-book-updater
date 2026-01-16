# Flashcard Generator for Obsidian Spaced Repetition

Create flashcards from the text section using the exact Obsidian Spaced Repetition plugin format. Extract key facts, definitions, formulas, arguments, and concepts worth memorizing.

## Format (use exactly)

One-way cards (how/why/process):
```
###### <Question>?
?
<Answer>
```

Bidirectional cards (facts/definitions):
```
###### <Question>?
??
<Answer>
```

## When to Use `?` vs `??`

- **Use `?`** for: "How"/"Why" questions, processes, explanations where the answer doesn't uniquely map back
- **Use `??`** for: Term ↔ Definition, Name ↔ Description, any case where both directions work as questions

## Content Rules

**Extract:**
- Key definitions (term and precise meaning)
- Core concepts and arguments
- Formulas, equations, and mathematical relationships
- Processes and sequential steps
- Important comparisons and distinctions
- Significant examples, data, and statistics

**Skip:**
- Trivial details, redundant points, pure narrative without conceptual weight

## Formatting Specifications

- **Questions**: `######` heading, clear and specific, one concept per card
- **Separator**: `?` or `??` on its own line
- **Answers**: Start immediately after separator; no blank lines within a card
- **Bold**: `**term**` for key terms in answers
- **Lists**: Bullet points or numbered lists are fine
- **LaTeX**: `$$...$$` for display math, `$...$` for inline
- **Tags**: Preserve `{preface}`, `{license}` exactly where they appear
- **Length**: Maximum 5-6 lines per answer; be concise but complete

## Answer Style

- **Definitions**: Do NOT include "X is..." or "X refers to..." (avoid circularity)
- **Structure**: Direct answer first, then brief context if helpful
- **Clarity**: Use simple language; avoid ambiguous phrasing

## Example

```markdown
###### What is the habit loop?
??
The three-step neurological pattern (cue, routine, reward) that creates automatic behaviors.

###### How do habits form neurologically?
?
Through repetition that shifts control from the cortex to the basal ganglia, reducing mental effort.

###### What brain region stores habits?
??
The **basal ganglia**.

###### What are the three components of the habit loop?
?
1. **Cue**: The trigger
2. **Routine**: The behavior
3. **Reward**: The reinforcement

###### What does the MIT rat maze study demonstrate?
?
That basal ganglia activity increases as behaviors become automatic, showing physical neural changes from repetition.

---

**Constraints**
- Output ONLY the flashcards (no headers, commentary, or explanatory text)
- Make sure there are blank lines between flashcards
- Preserve any `{tag}` elements exactly as they appear
