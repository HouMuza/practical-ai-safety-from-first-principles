from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE = Path(__file__).parents[1] / "src" / "safety_eval" / "mcq.py"
SPEC = importlib.util.spec_from_file_location("safety_eval_mcq", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CharacterTokenizer:
    def __init__(self) -> None:
        self.vocabulary = {character: index for index, character in enumerate(" abcABC:.")}

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [self.vocabulary[character] for character in text]}


class TransitionModel(torch.nn.Module):
    """Assigns logits from the current token to the next token."""

    def __init__(self, vocabulary_size: int, preferred: dict[int, int]) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.vocabulary_size = vocabulary_size
        self.preferred = preferred

    def forward(self, input_ids: torch.Tensor):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocabulary_size)
        for position in range(length):
            current = int(input_ids[0, position])
            preferred_next = self.preferred.get(current)
            if preferred_next is not None:
                logits[0, position, preferred_next] = 4.0
        return SimpleNamespace(logits=logits)


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()
        vocab = self.tokenizer.vocabulary
        self.model = TransitionModel(
            len(vocab),
            {
                vocab[":"]: vocab[" "],
                vocab[" "]: vocab["B"],
            },
        )

    def test_scores_option_tokens_only(self) -> None:
        result = MODULE.score_options(
            self.model,
            self.tokenizer,
            prompt="A:",
            options=[" A", " B"],
        )
        self.assertEqual(result.predicted_index_total, 1)
        self.assertEqual(result.predicted_index_mean, 1)
        self.assertEqual([score.token_count for score in result.scores], [2, 2])

    def test_rejects_empty_or_single_option_inputs(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.score_options(self.model, self.tokenizer, "", [" A", " B"])
        with self.assertRaises(ValueError):
            MODULE.score_options(self.model, self.tokenizer, "A:", [" A"])

    def test_single_token_path_uses_same_conditional_scores(self) -> None:
        generic = MODULE.score_options(
            self.model, self.tokenizer, prompt="A:", options=["A", "B"]
        )
        specialised = MODULE.score_single_token_options(
            self.model, self.tokenizer, prompt="A:", options=["A", "B"]
        )
        self.assertEqual(generic.predicted_index_total, specialised.predicted_index_total)
        self.assertEqual(
            [score.total_log_probability for score in generic.scores],
            [score.total_log_probability for score in specialised.scores],
        )

    def test_single_token_path_rejects_multi_token_option(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.score_single_token_options(
                self.model, self.tokenizer, prompt="A:", options=["A", "AB"]
            )

    def test_safetybench_prompt_is_stable(self) -> None:
        prompt, answers = MODULE.safetybench_prompt("Question?", ["Yes", "No"])
        self.assertEqual(
            prompt,
            "Question: Question?\nOptions:\n(A) Yes\n(B) No\nAnswer:(",
        )
        self.assertEqual(answers, ["A", "B"])

    def test_safetybench_chinese_prompt_is_stable(self) -> None:
        prompt, answers = MODULE.safetybench_prompt(
            "问题？", ["是", "否"], language="zh"
        )
        self.assertEqual(prompt, "问题：问题？\n选项：\n(A) 是\n(B) 否\n答案：(")
        self.assertEqual(answers, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
