# Puro-2B Research Programme

This folder contains research, experiments, evaluation code, and results related to Puro-2B. It is model-specific so additional models can have separate folders under `Chapter 17 - Bonus`.

Puro-2B is the first case study for the reusable [`safety-evaluation-pipeline`](../safety-evaluation-pipeline/). Model-independent code belongs in that sibling directory; this folder owns only Puro-specific research decisions and artefacts.

## Credit and independence

Puro-2B was created by Kairong Luo, Jiarui Cui, Yaorui Yin, Shengqi Chen, Yiming Yang, Linxiang Gao, Yanmohan Wang, Mingzhe Zhang, Kaiyue Wen, Kaifeng Lyu, and Wenguang Chen of the PACMAN Group at Tsinghua University. This independent safety-research project did not create Puro-2B and is not affiliated with or endorsed by its creators. See [ATTRIBUTION.md](ATTRIBUTION.md) for citations, source links, licences, and upstream acknowledgements.

## Structure

- `research-proposal.md`: the programme-level question, motivation, and roadmap.
- `model-inventory.md`: official checkpoints, training branches, artefacts, and experiment constraints.
- `ATTRIBUTION.md`: model credit, required citations, licences, and independence statement.
- `literature/`: literature review notes and the evidence table.
- `configs/`: shared model and evaluation configurations.
- `data/`: data documentation and small, redistributable research inputs. Large or restricted datasets should not be committed.
- `experiments/`: self-contained studies, numbered in the order they begin.
- `src/`: Puro-specific extensions only. Reusable code belongs in the chapter-level pipeline.
- `notebooks/`: exploratory analysis. Final results should be reproducible from scripts.
- `results/`: programme-level summary tables and figures.
- `paper/`: manuscripts and publication material.

## Experiment convention

Each experiment gets a directory such as `001-checkpoint-safety`. Its README records the research question, hypotheses, models, data, metrics, controls, procedure, analysis plan, status, and conclusions. Experiment-specific outputs stay inside that experiment; reusable code belongs in the chapter-level pipeline.

Do not overwrite raw results. Record exact model revisions, software versions, prompts, seeds, decoding settings, and hardware for every run.

## Active studies

- [`001-checkpoint-safety`](experiments/001-checkpoint-safety/): external
  SafetyBench baseline, pilot, and confirmatory checkpoint comparison.
- [`002-safety-boundary-stability`](experiments/002-safety-boundary-stability/):
  original paired challenge research derived from the book's treatment of
  distribution shift, calibration, red teaming, controls, and reproducibility.
