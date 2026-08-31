# Preregistration draft

This preregistration is shared across model-family bindings.

This document is deliberately incomplete. It becomes version 1.0 only after
scenario development and human review, and before any Puro checkpoint is scored.

## Decisions already fixed

- Three pinned Puro-2B endpoints and matched conditional-likelihood elicitation.
- Four semantic response roles with counterbalanced display positions.
- Scenario-family clustered analysis.
- Separation of unsafe, overcautious, and irrelevant flips.
- Non-operational content policy and two-reviewer approval.
- No reuse of development families in pilot or confirmatory evidence.

## Decisions required before the pilot freeze

- Required transformation matrix.
- Domain-by-use-context quotas.
- Number of development and pilot scenario families.
- Minimum inter-reviewer agreement and adjudication rule.
- Primary safe-margin scoring form: total versus length-normalised likelihood.
- Smallest practically important unsafe-flip increase.
- Exact bootstrap iterations and random seed.
- Missingness and tokenizer-failure policy.

## Decisions required before confirmation

- Confirmatory family count using pilot disagreement rates for feasibility only.
- Frozen primary interaction family and multiplicity rule.
- Independent author/reviewer separation between pilot and challenge sets.
- One additional model family for external replication, if compute permits.

## Outcome firewall

No outcome may be generated from draft or singly reviewed items. Development
results, once permitted, are method-debugging evidence and cannot be promoted
into pilot or confirmatory estimates.
