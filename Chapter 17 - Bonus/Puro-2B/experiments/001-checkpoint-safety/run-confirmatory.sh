#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "$experiment_dir/../../.." && pwd)"
pipeline_dir="$chapter_dir/safety-evaluation-pipeline"
dataset_path="$chapter_dir/Puro-2B/data/local/safetybench-en-zh.jsonl"
experiment_path="$experiment_dir/confirmatory-experiment.json"
sample_path="$experiment_dir/samples/confirmatory-4200.json"

cd "$pipeline_dir"

for model_alias in uniform curriculum_decay curriculum_sma6; do
  echo "[confirmatory] starting or resuming $model_alias"
  PYTHONPATH=src python3 -m safety_eval run \
    "$experiment_path" \
    --model "$model_alias" \
    --dataset "$dataset_path" \
    --profile sampled_accelerator \
    --sample-manifest "$sample_path"
done

echo "[confirmatory] all checkpoint runs complete"
