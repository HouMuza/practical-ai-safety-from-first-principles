# Scalable Safety-Evaluation Pipeline

This is the model-agnostic evaluation system for Chapter 17. Puro-2B is its first target, not a hard-coded dependency.

## Design goals

- Add a new model or checkpoint family through a versioned manifest.
- Keep benchmark logic independent of model-family logic.
- Support base, instruct, chat, reasoning, quantised, and API-served models through explicit adapters.
- Compare checkpoints with identical prompts, decoding, scoring, and metrics where scientifically valid.
- Preserve immutable model, dataset, code, environment, and output provenance.
- Separate smoke tests, pilots, and confirmatory runs.
- Resume interrupted runs without overwriting raw records.
- Make every reported table derivable from immutable raw outputs.

## Package boundaries

```text
safety-evaluation-pipeline/
├── configs/benchmarks/
├── docs/
├── schemas/
├── src/safety_eval/
└── tests/
```

Only the first tested scoring primitive and SafetyBench registry entry exist today. Additional modules should be added when a concrete experiment requires them.

The dependency-free local API can be started with `PYTHONPATH=src python3 -m
safety_eval.api`. It exposes `/api/health`, `/api/models`, and
`/api/experiments`; the chapter-level `../start.sh` starts it together with the
React dashboard.

## Portable CLI

The CLI has no mandatory third-party dependencies:

```bash
PYTHONPATH=src python3 -m safety_eval inspect-machine
PYTHONPATH=src python3 -m safety_eval profiles recommend
PYTHONPATH=src python3 -m safety_eval models
PYTHONPATH=src python3 -m safety_eval checks
PYTHONPATH=src python3 -m safety_eval plan \
  ../Puro-2B/experiments/001-checkpoint-safety/experiment.json \
  --profile smoke_cpu --max-items 3
```

Readers may install it in editable mode with `python3 -m pip install -e .` to
use the shorter `safety-eval` command. Model runtimes are optional extras, so a
CPU-only reader does not install a GPU stack merely to inspect or plan a run.

The planner resolves pinned model and check manifests, the observed machine,
the execution profile, output location, and item budget into a deterministic
run fingerprint. Planning never downloads model weights.

The shared runner accepts any conforming `ModelAdapter` and `SafetyCheck`. It
writes one append-only JSONL record per item, flushes each record to disk,
captures item failures, and resumes by skipping completed model/item pairs.
Failed items remain visible and are retried on a later invocation.

## Execute one model and check

Real execution is deliberately explicit. Supply one registered model alias and
a local SafetyBench JSON or JSONL file:

```bash
PYTHONPATH=src python3 -m safety_eval run \
  ../Puro-2B/experiments/001-checkpoint-safety/experiment.json \
  --model uniform \
  --dataset /path/to/safetybench-items.jsonl \
  --profile smoke_cpu \
  --max-items 20
```

By default, Transformers may use only files already in the local Hugging Face
cache. Add `--allow-download` only when downloading the pinned model and
tokenizer revision is intended. Every result records the dataset file SHA-256,
model revision, actual precision, quantisation, device, machine, and software.

SafetyBench input accepts a JSON list, `{ "items": [...] }`, `{ "data": [...]
}`, or one object per line in JSONL. Each item requires `question`, `options`,
and `answer`; `answer` may be a zero-based index or letter. Optional fields are
`id`/`item_id`, `language`, and `category`. The adapter generates a stable ID
when one is absent and samples round-robin across language/category strata.

## Freeze and analyse research samples

Freeze exact item membership before running models:

```bash
PYTHONPATH=src python3 -m safety_eval freeze-sample EXPERIMENT.json \
  --dataset /path/to/safetybench.jsonl \
  --max-items 70 \
  --evidence-class blinded_smoke \
  --output samples/blinded-smoke-70.json
```

Pass that manifest to every matched model with `run --sample-manifest ...`.
After all models have complete coverage, create a sanitized snapshot:

```bash
PYTHONPATH=src python3 -m safety_eval analyze \
  --records runs/RUN_ID/safetybench.jsonl \
  --experiment EXPERIMENT.json \
  --sample-manifest samples/SAMPLE.json \
  --output results/EXPERIMENT_ID
```

Analysis refuses paired conclusions when coverage is incomplete. It produces
Wilson intervals, paired bootstrap intervals, exact McNemar tests with Holm
adjustment, coverage and provenance manifests, CSV aggregates, limitations,
and a Markdown report. Pilot snapshots are labelled preliminary; only complete
confirmatory snapshots are marked publishable outcomes.

## Extension contract

A new model family supplies immutable model revisions and hashes, model stage, loading backend, precision, prompt policy, limits, licence metadata, and valid checkpoint comparison groups.

A benchmark supplies an immutable dataset revision and licence, task type, languages, prompt renderer, scoring method, metrics, uncertainty procedure, and contamination/redistribution notes.

The experiment config joins models and benchmarks. It must not modify their definitions silently.

Model-family manifests are validated against [`schemas/model-family.schema.json`](schemas/model-family.schema.json). The first conforming manifest is [`../Puro-2B/configs/models.json`](../Puro-2B/configs/models.json).
