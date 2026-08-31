"""Run neutral, non-research MCQs to validate one local Puro checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SOURCE = (
    Path(__file__).parents[3]
    / "safety-evaluation-pipeline"
    / "src"
    / "safety_eval"
    / "mcq.py"
)
SPEC = importlib.util.spec_from_file_location("safety_eval_mcq", SOURCE)
SCORING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCORING
assert SPEC.loader is not None
SPEC.loader.exec_module(SCORING)


NEUTRAL_ITEMS = [
    {"id": "arithmetic", "question": "What is 2 + 2?", "options": ["3", "4", "5"], "answer": 1},
    {
        "id": "geography",
        "question": "What is the capital of France?",
        "options": ["Berlin", "Paris", "Rome"],
        "answer": 1,
    },
    {
        "id": "sequence",
        "question": "Which number comes next: 2, 4, 6, 8?",
        "options": ["9", "10", "12"],
        "answer": 1,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    load_seconds = time.perf_counter() - started

    results = []
    scoring_started = time.perf_counter()
    for item in NEUTRAL_ITEMS:
        prompt, answer_options = SCORING.safetybench_prompt(
            item["question"], item["options"]
        )
        score = SCORING.score_single_token_options(
            model, tokenizer, prompt, answer_options
        )
        results.append(
            {
                "id": item["id"],
                "expected_index": item["answer"],
                **score.to_dict(),
            }
        )

    report = {
        "model_path": str(args.model_path),
        "torch_version": torch.__version__,
        "device": "cpu",
        "dtype": "bfloat16",
        "load_seconds": load_seconds,
        "scoring_seconds": time.perf_counter() - scoring_started,
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
