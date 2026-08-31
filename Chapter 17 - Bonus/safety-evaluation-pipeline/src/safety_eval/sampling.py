"""Freeze deterministic check samples before model outcomes are inspected."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def item_ids_sha256(item_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(item_ids).encode()).hexdigest()


def freeze_sample(
    check: Any,
    *,
    experiment_id: str,
    seed: int,
    max_items: int,
    evidence_class: str,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen sample: {output_path}")
    items = list(check.items(seed=seed, max_items=max_items))
    if len(items) != max_items:
        raise ValueError(f"Requested {max_items} items but the check supplied {len(items)}")
    item_ids = [str(item["item_id"]) for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("The selected sample contains duplicate item IDs")
    strata = Counter(f"{item.get('language', 'unknown')}/{item.get('category', 'unknown')}" for item in items)
    manifest = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "check": {key: value for key, value in dict(check.identity).items() if key != "dataset_path"},
        "evidence_class": evidence_class,
        "seed": seed,
        "sampling_method": "deterministic_stratified_round_robin",
        "item_count": len(item_ids),
        "item_ids": item_ids,
        "item_ids_sha256": item_ids_sha256(item_ids),
        "strata": dict(sorted(strata.items())),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_sample(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    item_ids = manifest.get("item_ids")
    if not isinstance(item_ids, list) or not item_ids:
        raise ValueError("Sample manifest has no item IDs")
    if item_ids_sha256([str(item) for item in item_ids]) != manifest.get("item_ids_sha256"):
        raise ValueError("Sample manifest item-ID hash does not match")
    return manifest


def attach_sample_to_plan(plan: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    if sample["experiment_id"] != plan["experiment"]["experiment_id"]:
        raise ValueError("Sample manifest belongs to another experiment")
    digest = hashlib.sha256(
        f"{plan['fingerprint_sha256']}:{sample['item_ids_sha256']}".encode()
    ).hexdigest()
    updated = dict(plan)
    updated["sample"] = {key: value for key, value in sample.items() if key != "item_ids"}
    updated["max_items"] = sample["item_count"]
    updated["fingerprint_sha256"] = digest
    updated["run_id"] = f"{sample['experiment_id']}-{digest[:12]}"
    return updated
