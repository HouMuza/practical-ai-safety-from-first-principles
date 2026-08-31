"""Pre-outcome validation for authored safety-boundary scenario suites."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_boundary_items(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text) if path.suffix == ".json" else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(payload, list):
        raise ValueError("Boundary suite must be a JSON array or JSONL records")
    return payload


def validate_boundary_suite(path: Path, *, require_approved: bool = False) -> dict[str, Any]:
    items = load_boundary_items(path)
    if not items:
        raise ValueError("Boundary suite is empty")
    ids = [str(item.get("item_id")) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Boundary suite contains duplicate item IDs")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item.get("scenario_id"))].append(item)
    problems = []
    for scenario_id, family in groups.items():
        clean = [item for item in family if item.get("transformation") == "clean"]
        if len(clean) != 1:
            problems.append(f"{scenario_id}: expected exactly one clean anchor")
        variants = [str(item.get("variant_id")) for item in family]
        if len(variants) != len(set(variants)):
            problems.append(f"{scenario_id}: duplicate variant IDs")
        if len(family) < 2:
            problems.append(f"{scenario_id}: needs at least one transformed variant")
    if require_approved:
        for item in items:
            review = item.get("review") or {}
            if review.get("status") != "approved" or review.get("label_validity") != "pass" or review.get("operational_detail_safe") != "pass":
                problems.append(f"{item.get('item_id')}: approval gates incomplete")
            if item.get("transformation") != "clean" and review.get("semantic_preservation") != "pass":
                problems.append(f"{item.get('item_id')}: semantic preservation not approved")
            if len(set(review.get("reviewer_ids") or [])) < 2:
                problems.append(f"{item.get('item_id')}: fewer than two independent reviewers")
    if problems:
        raise ValueError("Boundary suite validation failed:\n- " + "\n- ".join(problems[:20]))
    answer_positions = Counter(
        next(index for index, option in enumerate(item["options"]) if option["option_id"] == "safe_helpful")
        for item in items
    )
    return {
        "items": len(items),
        "scenario_families": len(groups),
        "domains": dict(sorted(Counter(str(item["domain"]) for item in items).items())),
        "transformations": dict(sorted(Counter(str(item["transformation"]) for item in items).items())),
        "languages": dict(sorted(Counter(str(item["language"]) for item in items).items())),
        "use_contexts": dict(sorted(Counter(str(item["use_context"]) for item in items).items())),
        "safe_answer_positions": {str(key): value for key, value in sorted(answer_positions.items())},
        "approval_required": require_approved,
    }
