"""Stable extension contracts for model backends and safety checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def score_options(self, prompt: str, options: list[str]) -> Mapping[str, Any]: ...

    def generate(self, prompt: str, **generation: Any) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


@runtime_checkable
class SafetyCheck(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def items(self, *, seed: int, max_items: int | None) -> Iterable[Mapping[str, Any]]: ...

    def render(self, item: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def score(self, item: Mapping[str, Any], model_output: Mapping[str, Any]) -> Mapping[str, Any]: ...
