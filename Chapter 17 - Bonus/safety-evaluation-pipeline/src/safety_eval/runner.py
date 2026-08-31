"""Append-only, resumable execution engine independent of model/check plugins."""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .interfaces import ModelAdapter, SafetyCheck


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _software() -> dict[str, str]:
    return {"python": platform.python_version(), "platform": platform.platform(), "implementation": sys.implementation.name}


def completed_item_ids(path: Path, model_alias: str, run_id: str) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as records:
        for line_number, line in enumerate(records, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if record.get("run_id") != run_id:
                raise ValueError(f"Refusing to mix run IDs in {path}")
            if record.get("status") == "completed" and record.get("model", {}).get("model_alias") == model_alias:
                completed.add(record["item_id"])
    return completed


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def run_check(
    plan: dict[str, Any],
    model: ModelAdapter,
    check: SafetyCheck,
    *,
    records_path: Path,
) -> dict[str, int]:
    model_identity = dict(model.identity)
    check_identity = dict(check.identity)
    model_alias = str(model_identity["model_alias"])
    already_complete = completed_item_ids(records_path, model_alias, plan["run_id"])
    counts = {"completed": 0, "failed": 0, "resumed": len(already_complete)}
    items = check.items(seed=plan["experiment"]["seed"], max_items=plan["max_items"])

    try:
        for item in items:
            item_id = str(item["item_id"])
            if item_id in already_complete:
                continue
            started_at = _now()
            rendered: dict[str, Any] | None = None
            try:
                rendered = dict(check.render(item))
                if "options" in rendered:
                    model_output = dict(model.score_options(str(rendered["prompt"]), list(rendered["options"])))
                else:
                    model_output = dict(model.generate(str(rendered["prompt"])))
                scored = dict(check.score(item, model_output))
                status = "completed"
                error = None
                counts["completed"] += 1
            except Exception as exception:  # item failures are data, not a lost run
                model_output = None
                scored = None
                status = "failed"
                error = {"type": type(exception).__name__, "message": str(exception)}
                counts["failed"] += 1
            append_record(
                records_path,
                {
                    "schema_version": "1.0",
                    "run_id": plan["run_id"],
                    "experiment_id": plan["experiment"]["experiment_id"],
                    "item_id": item_id,
                    "model": model_identity,
                    "check": check_identity,
                    "environment": {"machine": plan["machine"], "software": _software(), "execution_profile": plan.get("profile")},
                    "status": status,
                    "started_at": started_at,
                    "completed_at": _now(),
                    "input": rendered,
                    "output": {"model": model_output, "score": scored} if status == "completed" else None,
                    "error": error,
                },
            )
    finally:
        model.close()
    return counts
