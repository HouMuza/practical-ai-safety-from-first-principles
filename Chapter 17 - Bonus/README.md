# Chapter 17 — Bonus: Open Model Safety Evaluations

This chapter develops a reusable safety-evaluation pipeline and applies it to open model families with sufficiently transparent artefacts.

## Separation of concerns

- [`safety-evaluation-pipeline`](safety-evaluation-pipeline/): model-agnostic loading, benchmark adapters, scoring, metrics, provenance, schemas, and tests.
- [`reporting-dashboard`](reporting-dashboard/): React frontend for experiment comparisons, research status, methodology, and provenance.
- [`Puro-2B`](Puro-2B/): the first model-family study, including its hypotheses, checkpoint manifests, literature, experiment protocols, and results.

Future model families should receive sibling directories such as `Nemotron/`. They reuse the shared pipeline and contribute only model manifests, study bindings, adapters when genuinely necessary, and results. Protocols live in the model-independent [`research-studies`](research-studies/) registry and are never copied per model.

The registry maps every executable experiment in Chapters 2-16 to its research question, required interfaces, reference notebook, and development/pilot/confirmatory compute stages. A CPU reader can still validate schemas and run toy experiments; a larger machine unlocks model inference, training, multi-seed and full-dataset evidence.

The pipeline is the book's reusable technical contribution. Individual model folders are case studies produced with it.

## Run the local platform

From this directory, start the local API and React dashboard together:

```bash
./start.sh
```

The dashboard runs at `http://localhost:3000`; the API health endpoint is
`http://localhost:8788/api/health`. Use `DASHBOARD_PORT` and `SAFETY_API_PORT`
to override those defaults. Press Ctrl+C to stop both services.

The local API currently reads versioned research manifests and returns JSON. It
does not require a database. Experiment outputs remain immutable files so they
can be reproduced, reviewed, and committed with the study.
