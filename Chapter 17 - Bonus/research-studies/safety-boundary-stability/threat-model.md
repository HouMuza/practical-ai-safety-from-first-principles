# Threat model

This threat model applies to every checkpoint bound to the study.

## System under study

The primary systems are the three pinned Puro-2B base-model checkpoints scored
by conditional likelihood over four response policies. This study evaluates
model knowledge and preference under a controlled elicitation procedure; it
does not emulate a deployed assistant with system prompts or guardrails.

## Safety failure

A failure occurs when a meaning-preserving transformation changes a previously
safe and useful judgement into an unsafe, irrelevant, or unnecessarily
overcautious choice. The failure is important even before a hard choice flips
when the likelihood margin assigned to the safe response degrades materially.

## Actors and shifts

- Ordinary users create natural variation through language, register,
  paraphrase, and incomplete context.
- Benign specialists use technical vocabulary that must not be mistaken for
  harmful intent.
- A weak black-box adversary may reframe a request after observing whether a
  system accepts or rejects it, but receives no gradients or model internals.
- Translation and bilingual use introduce semantic and cultural variation.

The confirmatory study will not claim robustness to adaptive optimisation,
multi-turn attacks, hidden system prompts, tool use, or white-box attacks.

## Protected properties

- Harmful intent should not become acceptable under superficial reframing.
- Legitimate use should remain answerable rather than triggering blanket
  refusal.
- Irrelevant context should not move the safety boundary.
- Equivalent English and Chinese scenarios should receive comparable
  judgements after human semantic review.

## Misuse and researcher-safety boundary

Scenario text must remain non-operational. Unsafe answer options may describe
an inappropriate policy at a high level but must not provide procedural details
that increase a reader's ability to cause harm. Reviewers can reject any item
without preserving it for public release.
