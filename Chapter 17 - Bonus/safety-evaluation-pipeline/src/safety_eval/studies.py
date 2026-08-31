"""Discover book studies and explain which evidence stages a machine can run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hardware import Machine, inspect_machine


PIPELINE_DIR = Path(__file__).resolve().parents[2]
CHAPTER_DIR = PIPELINE_DIR.parent
CATALOG_PATH = CHAPTER_DIR / "research-studies" / "catalog.json"


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        catalog = json.load(source)
    identifiers = [study["study_id"] for study in catalog["studies"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Study IDs must be unique")
    return catalog


def machine_capabilities(machine: Machine) -> set[str]:
    capabilities = {"cpu"}
    memory_gib = (machine.memory_bytes or 0) / (1024**3)
    kinds = {accelerator.kind for accelerator in machine.accelerators}
    if kinds:
        capabilities.add("accelerator")
    capabilities.update(kinds)
    if memory_gib >= 24:
        capabilities.add("high_memory")
    if sum(1 for accelerator in machine.accelerators if accelerator.kind == "cuda") >= 2:
        capabilities.add("multi_accelerator")
    return capabilities


def plan_study(study_id: str, *, machine: Machine | None = None, catalog_path: Path = CATALOG_PATH) -> dict[str, Any]:
    observed = machine or inspect_machine()
    catalog = load_catalog(catalog_path)
    try:
        study = next(item for item in catalog["studies"] if item["study_id"] == study_id)
    except StopIteration as error:
        raise ValueError(f"Unknown study {study_id!r}") from error
    capabilities = machine_capabilities(observed)
    stages = []
    for stage in study["stages"]:
        missing = sorted(set(stage["requires"]) - capabilities)
        stages.append({**stage, "ready": not missing, "missing_capabilities": missing})
    runnable = [stage for stage in stages if stage["ready"]]
    return {
        "schema_version": "1.0",
        "study": {key: value for key, value in study.items() if key != "stages"},
        "machine": observed.to_dict(),
        "machine_capabilities": sorted(capabilities),
        "stages": stages,
        "recommended_stage": runnable[-1]["stage_id"] if runnable else None,
        "warning": "Compatibility is not an outcome. Freeze a model binding and protocol before collecting evidence."
    }
