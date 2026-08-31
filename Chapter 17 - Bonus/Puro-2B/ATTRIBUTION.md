# Attribution and Provenance

## Puro-2B creators

Puro-2B and its open pretraining recipe were created by the PACMAN Group at Tsinghua University. The authors of the original technical report are:

- Kairong Luo
- Jiarui Cui
- Yaorui Yin
- Shengqi Chen
- Yiming Yang
- Linxiang Gao
- Yanmohan Wang
- Mingzhe Zhang
- Kaiyue Wen
- Kaifeng Lyu
- Wenguang Chen

Primary work:

> Kairong Luo, Jiarui Cui, Yaorui Yin, Shengqi Chen, Yiming Yang, Linxiang Gao, Yanmohan Wang, Mingzhe Zhang, Kaiyue Wen, Kaifeng Lyu, and Wenguang Chen. “Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090.” arXiv:2608.27370, 2026.

- Paper: <https://arxiv.org/abs/2608.27370>
- Official model: <https://huggingface.co/thu-pacman/Puro-2B-Base>
- Official model collection: <https://huggingface.co/collections/thu-pacman/puro-2b>
- Official training repository: <https://github.com/thu-pacman/Puro-Megatron>

Use the BibTeX record in [`literature/references.bib`](literature/references.bib) when citing the model or training recipe in academic material.

## Upstream software

Puro-Megatron is described by its creators as a focused patch series on top of NVIDIA Megatron-LM/Megatron Core `core_v0.16.0`, forked at commit `3bec9aa97dda898d16ff5a89bac0ed2b6682b172`. Work using that training implementation should also preserve its licence notices and cite Megatron-LM as requested by the upstream repository.

The references file includes the original Megatron-LM paper. Any additional evaluation libraries, datasets, benchmarks, model judges, or derived checkpoints used in our experiments must be added to this document and the references file before results are published.

## Licence handling

At the time this record was prepared, the official `thu-pacman/Puro-2B-Base` model card identified the model licence as Apache-2.0, and the Puro-Megatron repository included a licence file. A model card label is not a substitute for checking every artefact we use.

For each experiment, record:

- the exact model or dataset repository;
- the immutable revision or commit hash;
- the licence present at that revision;
- the date it was retrieved;
- any attribution, notice, redistribution, or acceptable-use requirements;
- whether weights, data, prompts, or outputs may be redistributed.

Copies or modifications of licensed material must retain all notices required by the applicable licence. This repository's own licence, if one is added, does not replace or override third-party licences.

## Independence statement

This directory contains independent follow-on safety research. Its authors did not create Puro-2B or the Puro-Megatron training recipe. Unless explicitly documented otherwise, this work is not affiliated with, sponsored by, reviewed by, or endorsed by the Puro-2B authors, the PACMAN Group, Tsinghua University, NVIDIA, or other upstream contributors.

Public descriptions should say “research using/evaluating Puro-2B,” not “our Puro model.” Findings, errors, interpretations, and derived artefacts from this project are our responsibility and should not be attributed to the original creators.

## Research reporting checklist

Every experiment and publication must:

1. Cite the Puro-2B technical report.
2. Link to the exact official model/checkpoint and record its revision.
3. Credit the PACMAN Group and named paper authors.
4. Cite and comply with upstream software, datasets, and benchmarks.
5. Clearly separate facts reported by the creators from our reproduction results and interpretations.
6. Label any modified or derived checkpoint so it cannot be mistaken for an official release.
7. Include the independence statement or an equivalent disclosure.

## Corrections

Attribution metadata can change as a new project matures. Before any public release, compare this record with the latest official paper, model cards, repository notices, and dataset documentation. Correct omissions promptly and preserve a dated record of the change.
