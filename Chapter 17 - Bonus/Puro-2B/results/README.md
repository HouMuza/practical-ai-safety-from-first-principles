# Programme Results

Store cross-experiment summary tables and figures here. Raw and experiment-specific results remain with their experiment and should not be overwritten.
# Results

Sanitized aggregate snapshots live here. Raw JSONL records remain under the
corresponding experiment's ignored `runs/` directory.

`001-checkpoint-safety/blinded-smoke-70/` records complete matched coverage for
all three Puro-2B checkpoints. Because this was a blinded smoke stage, accuracy
and pairwise outcomes are deliberately withheld. The snapshot contains only
coverage, provenance, limitations, and reporting-status information.

`001-checkpoint-safety/pilot-700/` is the sanitized snapshot from the
preregistered 700-item pilot. It contains aggregate accuracy and Wilson
intervals, category and language summaries, paired bootstrap intervals, exact
McNemar tests with Holm correction, provenance, and limitations. It contains no
benchmark prompts, item IDs, or item-level answers. Its status is
`preliminary`, and `publishable_outcome` remains false until a distinct
confirmatory protocol is frozen and completed.
