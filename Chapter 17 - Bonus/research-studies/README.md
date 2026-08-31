# Book research study registry

This directory is the model-independent research layer for *AI Safety from First
Principles*. A study defines the question, intervention, measurements, evidence
artefacts and compute stages once. Model-family folders contain only bindings
that say which checkpoints fill the study's subject roles.

The existing chapter notebooks remain the reference executors while their logic
is progressively extracted into reusable pipeline adapters. They are not copied
into each model folder.

```text
research-studies/catalog.json     all executable book studies
research-studies/<study>/         shared protocol, data and preregistration
Puro-2B/studies/                   Puro-specific bindings only
safety-evaluation-pipeline/       schemas, planners, runners and result records
```

List the catalogue and see what this machine can run:

```bash
cd "safety-evaluation-pipeline"
python -m safety_eval studies
python -m safety_eval study-plan ch10-reward-hacking
```

`ready` means the current machine can execute that evidence stage. It does not
mean that the study has been run or that an outcome exists. Runs must still pin
models, data, prompts, seeds and environment details and must write append-only
records.

