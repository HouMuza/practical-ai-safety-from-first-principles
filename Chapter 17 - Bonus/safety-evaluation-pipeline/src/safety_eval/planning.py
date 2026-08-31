"""Resolve a portable experiment manifest into an explicit execution plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .hardware import Machine, inspect_machine, load_profiles, recommend_profile


PIPELINE_DIR = Path(__file__).resolve().parents[2]
CHAPTER_DIR = PIPELINE_DIR.parent
PROFILES_PATH = PIPELINE_DIR / "configs" / "execution-profiles.json"


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _select_profile(profile_id: str, machine: Machine) -> dict[str, Any]:
    profiles = load_profiles(PROFILES_PATH)
    if profile_id == "auto":
        return recommend_profile(machine, profiles)
    try:
        return next(profile for profile in profiles if profile["profile_id"] == profile_id)
    except StopIteration as error:
        available = ", ".join(profile["profile_id"] for profile in profiles)
        raise ValueError(f"Unknown profile {profile_id!r}; choose one of: {available}, auto") from error


def resolve_experiment(
    experiment_path: Path,
    *,
    profile_override: str | None = None,
    max_items_override: int | None = None,
    machine: Machine | None = None,
) -> dict[str, Any]:
    experiment_path = experiment_path.resolve()
    experiment = _load(experiment_path)
    observed_machine = machine or inspect_machine()
    profile = _select_profile(profile_override or experiment["profile"], observed_machine)

    resolved_models = []
    for requested in experiment["models"]:
        manifest_path = CHAPTER_DIR / requested["family_id"].replace("_", "-") / "configs" / "models.json"
        if not manifest_path.exists():
            candidates = list(CHAPTER_DIR.glob("*/configs/models.json"))
            manifest_path = next(
                (path for path in candidates if _load(path).get("family_id") == requested["family_id"]),
                manifest_path,
            )
        manifest = _load(manifest_path)
        try:
            model = next(item for item in manifest["models"] if item["alias"] == requested["model_alias"])
        except StopIteration as error:
            raise ValueError(f"Model alias {requested['model_alias']!r} is not in {manifest_path}") from error
        resolved_models.append({"family_id": manifest["family_id"], **model})

    resolved_checks = []
    for check_id in experiment["checks"]:
        check_path = PIPELINE_DIR / "configs" / "benchmarks" / f"{check_id}.json"
        if not check_path.exists():
            raise ValueError(f"Safety check {check_id!r} is not registered")
        resolved_checks.append(_load(check_path))

    max_items = max_items_override
    if max_items is None:
        configured_max_items = experiment.get("max_items_override")
        max_items = configured_max_items if configured_max_items is not None else profile["max_items"]

    fingerprint_source = {
        "experiment": experiment,
        "models": resolved_models,
        "checks": resolved_checks,
        "profile": profile,
        "max_items": max_items,
        "machine": observed_machine.to_dict(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_source, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_directory = (experiment_path.parent / experiment.get("output_directory", "runs")).resolve()
    return {
        "schema_version": "1.0",
        "run_id": f"{experiment['experiment_id']}-{fingerprint[:12]}",
        "fingerprint_sha256": fingerprint,
        "experiment": experiment,
        "models": resolved_models,
        "checks": resolved_checks,
        "profile": profile,
        "max_items": max_items,
        "machine": observed_machine.to_dict(),
        "output_directory": str(output_directory),
    }
