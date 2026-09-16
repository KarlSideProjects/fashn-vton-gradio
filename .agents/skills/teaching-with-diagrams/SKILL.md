---
name: teaching-with-diagrams
description: Create beginner-friendly technical lessons, theory explanations, or learning-oriented READMEs with Mermaid diagrams, grounded examples and practice. Use when teaching a technical topic or turning a codebase or paper into learning materials; also supports checking updates to this skill's three upstream sources.
license: MIT
---

# Teaching with Diagrams

Combine learning design, Diátaxis document structure and Mermaid diagrams into the user's requested deliverable. This is a self-contained adaptation of `teach`, `documentation-writer` and `mermaid-diagrams`; installing or invoking those skills separately is unnecessary. Their exact reviewed versions are in [upstream.lock.json](upstream.lock.json). Read [composition and maintenance](references/composition.md) when evaluating upstream changes or the reasons for these adaptations.

## Establish the learning outcome

Infer the audience, existing knowledge, language, practical goal and output format from the conversation and selected project. Ask only for missing information that materially changes the lesson. When asked to write, use an inferred outline and complete the document unless the user requests outline approval.

Choose the deliverable's purpose:

- **Explanation:** build an understanding of why something works, with intuition and a worked example.
- **Tutorial:** guide a beginner to a small, observable success, including expected results and recovery from likely mistakes.
- **How-to:** solve a particular task for a reader who already knows the basics.
- **Reference:** provide exact parameters, definitions and constraints for lookup.

A learning README can lead with explanation and a short tutorial, then link to existing setup and reference documents. Keep useful installation and operation instructions discoverable. Follow the requested Markdown or HTML format. Create a multi-session course and learning records only when ongoing learning is requested; record demonstrated learning, not assumed mastery.

## Ground the content

Read the relevant source code, supplied material, official documentation or original papers before making implementation-specific claims. Follow the environment's source-discovery tools and policies. Prefer versioned links for code and record the relevant model or library version.

Distinguish general theory, verified implementation, experimental observations and open hypotheses. A single successful example supports a case-specific observation. Explain an analogy's limits and define every symbol before using an equation. Missing sources are an explicit evidence gap, never a reason to invent details.

## Teach one step at a time

Start with the learner's concrete problem. Introduce only the prerequisites needed for the next step. Define technical terms beside a familiar example, then connect intuition to the actual mechanism. For quantitative topics, walk through a small numerical example before the full formula.

Build a short practice or prediction task where it helps the learning goal. Provide expected observations, an answer or feedback, and the reason behind it; let a reader check their understanding without waiting for another chat. For experimental topics, change one variable at a time and explain what the result can and cannot establish. Keep optional deeper theory separate from the main path.

## Explain relationships with Mermaid

Use a diagram when it makes a process, dependency or distinction easier to understand. A request explicitly asking for Mermaid should include a relevant Mermaid diagram. Read [diagram guidance](references/mermaid.md) before authoring or revising diagrams.

Each diagram should answer one question. Introduce what to look for, provide the diagram, then explain its key relationship in prose. Keep the Mermaid source in the requested document so readers can edit it. Use a flowchart for transformations, a sequence diagram for interactions over time, and separate small diagrams for different levels of detail.

Validate syntax with an available local Mermaid renderer and inspect the rendered result when possible. Report syntax validation and visual inspection separately. If rendering is unavailable, provide the source and explain that rendering remains unverified. Readable prose must still convey the diagram's meaning.

## Finish the deliverable

Verify references and internal links, example arithmetic, diagram labels and consistency with the cited implementation. Check that practice includes feedback and that operating instructions still work with the document structure. Report the resulting files and actual validation; do not claim the learner has mastered a topic merely because the document was generated.

## Check upstream updates

On an update-check request, resolve this installed skill directory and run:

```bash
node <skill-directory>/scripts/check-upstreams.mjs
```

This requires Node.js 22+ and network access to GitHub. It checks all three sources and their licenses against the recorded baseline, prints JSON, and changes no files. Exit codes: `0` unchanged, `2` updates found, `1` one or more checks failed. Use [composition and maintenance](references/composition.md) to review changes and intentionally refresh the baseline. Teaching requests use the installed guidance without a network update check.
