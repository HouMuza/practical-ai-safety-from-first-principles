"""Reusable safety-evaluation components for Chapter 17.

Inference dependencies are lazy so planning commands work without PyTorch.
"""

__all__ = ["MultipleChoiceResult", "OptionScore", "safetybench_prompt", "score_options", "score_single_token_options"]


def __getattr__(name: str):
    if name in __all__:
        from . import mcq

        return getattr(mcq, name)
    raise AttributeError(name)
