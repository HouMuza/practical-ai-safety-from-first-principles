# Experiment 002: Safety Boundary Stability (legacy wrapper)

The shared protocol, scenarios, preregistration, threat model and review rubric
now live in [`../../../research-studies/safety-boundary-stability`](../../../research-studies/safety-boundary-stability/).
Puro's subject selection lives in
[`../../studies/safety-boundary-stability.binding.json`](../../studies/safety-boundary-stability.binding.json).
This directory keeps its original manifest temporarily for old runner commands;
do not copy it when adding another model family.

## Status

Instrument development. No model outcomes have been generated.

## Original research question

Do matched Puro-2B training recipes produce safety judgements that remain
stable when meaning is preserved but language, register, framing, or irrelevant
context changes?

This is a book-derived research instrument, not a reproduction of an existing
leaderboard. Existing benchmarks may be used only as external context or
capability controls.

## Research artefacts

- [`threat-model.md`](threat-model.md) defines the system, failure, adversary,
  and claim boundary.
- [`protocol.md`](protocol.md) defines the paired design and estimands.
- [`preregistration-draft.md`](preregistration-draft.md) lists decisions that
  must be frozen before the pilot.
- [`scenario-authoring-guide.md`](scenario-authoring-guide.md) defines how
  scenario families and transformations are written.
- [`human-review-rubric.md`](human-review-rubric.md) prevents invalid
  transformations from being counted as model failures.
- [`experiment.json`](experiment.json) registers the three Puro checkpoints and
  the reusable `safety_boundary_stability` check.
- [`data/scenario-template.json`](data/scenario-template.json) is an authoring
  template, not evaluation data.
- [`data/development-briefs.json`](data/development-briefs.json) contains the
  first 14 non-operational authoring briefs, balanced across seven domains. It
  is a backlog, not a labelled or approved evaluation set.

## Evidence ladder

1. Schema and scorer validation with synthetic fixtures.
2. Human-reviewed development families used only to debug the instrument.
3. Blinded pilot families used to estimate disagreement and review burden.
4. Independently authored and frozen confirmatory challenge families.
5. Replication on another model family before broadening the claim.
