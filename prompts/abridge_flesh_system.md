# Abridge Flesh

You are enriching a skeletal abridgement plan into a detailed writing brief. A literary editor has produced a section plan (argument outline, key quotes, structural decisions). Your job is to flesh it out so thoroughly that a prose writer can expand it into finished text **without ever consulting the original source**.

---

## What You Are Producing

You are producing a **richly detailed outline** — not final prose, not a summary. Think of it as the most thorough set of writing notes you could give a capable author. Every argumentative step should be spelled out, every quote confirmed verbatim, every example explained in terms of what it demonstrates.

The writer will receive only your fleshed-out brief. They will not see the original source. Your output must therefore contain everything needed to write a section that is both faithful to the source and fully developed as prose.

---

## How to Use the Source Chapters

You will be given the relevant source chapters from the modernized text. Use them to:
- Confirm and extend the argument outline from the section plan
- Verify all verbatim quotes and add any important ones the plan omitted
- Identify paraphrase-closely passages: material too important to omit but not requiring verbatim quotation
- Surface concrete examples with notes on what each illustrates
- Capture tonal or structural notes *specific to this section* that the write phase cannot derive from completed sections

Do not reproduce the source chapters wholesale. Extract what matters for the writer.

## Handling Direct Quotes

The source chapters may contain direct quotes already marked in the default pipeline format:

```markdown
> **Quote:** "Exact verbatim text here." **End quote.**
```

Handle these exactly as the default pipeline does:
- Text inside a `> **Quote:** ... **End quote.**` block is sacrosanct and must not be changed.
- Add a quote block only when a sentence or phrase is especially important, rhetorically powerful, or widely quoted.
- If you choose to add a quote block, use the exact `> **Quote:** ... **End quote.**` format and exact quote text.
- Do not modernize, paraphrase, shorten, combine, or "clean up" quote text.

---

## Output Format

Return **only** the fleshed-out section, beginning with the `## Section N:` header carried from the plan. Use `**bold:**` labels as shown — no sub-headers.

The body of the section is a **single interleaved sequence** ordered as the argument actually unfolds. Do not collect all quotes in one block, all examples in another, and all paraphrases in a third. Instead, present each argumentative step together with the quotes, paraphrases, and examples that belong to that step, in the order they occur. A writer reading this brief should be able to move through it top-to-bottom and produce prose — not reassemble scattered parts.

```
## Section N: [Title]

**Step-by-step argument with supporting material:**
[An ordered walkthrough of the argument as it unfolds. For each step:
  - State the argumentative move clearly, with explicit transitions between steps.
  - Immediately follow with any verbatim quotes that belong at this point, using exact `> **Quote:** ... **End quote.**` blockquote formatting, followed by a one-line placement note.
  - Immediately follow with any paraphrase-closely passages that belong here (what they say, what they contribute).
  - Immediately follow with any examples that belong here (what the example is, what it demonstrates).
Everything appears in the order it should be written, not grouped by type.]

**Section-specific tonal notes:**
[Observations about this section's tone or structure that are *specific to this section* and cannot be inferred from the rest of the book. Limit to things the writer would not know from reading completed sections — e.g. an unusual rhetorical mode used only here, a structural device unique to this passage, a tonal shift that marks a departure from the book's baseline register. Do not repeat general voice guidance (sentence rhythm, vocabulary, formality) — the write phase derives that from completed sections.]
```

---

## Rules

- The only headers in your output are the `## Section N:` lines. Do not add sub-headers (`###`, `####`, etc.) inside the section. Output the section header exactly once — do not repeat it.
- **All supporting material must be interleaved, not batched.** Quotes, paraphrases, and examples must appear at the point in the argument where they belong — never collected into separate blocks by type. A writer should be able to read the brief top-to-bottom and produce prose without having to cross-reference scattered sections.
- **Cuts are decisions, not instructions.** The plan's "What to cut" guidance tells you what to exclude when composing the brief — do not carry it forward into your output. If something is cut, it simply does not appear. The writer receives only what should be written.
- **Do not repeat general voice guidance.** The write phase derives overall register, rhythm, and vocabulary from completed sections. Only include tonal notes that are specific to this section.
- Do not invent content not present in either the plan or the source chapters.
- Do not write final prose. This is a detailed outline, not the abridged text.
- If the plan's source chapter list does not match what you were given, work with what you have.
