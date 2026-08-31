"""Sanitized aggregate analysis for matched safety-evaluation runs."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .sampling import load_sample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wilson(correct: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _metric(values: list[bool]) -> dict[str, Any]:
    correct = sum(values)
    low, high = _wilson(correct, len(values))
    return {"n": len(values), "correct": correct, "accuracy": correct / len(values) if values else None, "ci95_wilson": [low, high]}


def _bootstrap_difference(a: list[bool], b: list[bool], *, seed: int, iterations: int) -> list[float]:
    generator = random.Random(seed)
    differences = []
    for _ in range(iterations):
        indices = [generator.randrange(len(a)) for _ in a]
        differences.append(sum(a[index] for index in indices) / len(a) - sum(b[index] for index in indices) / len(b))
    differences.sort()
    return [differences[int(0.025 * (iterations - 1))], differences[int(0.975 * (iterations - 1))]]


def _mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    a_only = sum(left and not right for left, right in zip(a, b))
    b_only = sum(right and not left for left, right in zip(a, b))
    discordant = a_only + b_only
    if discordant == 0:
        return a_only, b_only, 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / (2**discordant)
    return a_only, b_only, min(1.0, 2 * tail)


def _holm(comparisons: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(comparisons), key=lambda item: item[1]["mcnemar_p_value"])
    running = 0.0
    count = len(ordered)
    for rank, (original_index, comparison) in enumerate(ordered):
        adjusted = min(1.0, comparison["mcnemar_p_value"] * (count - rank))
        running = max(running, adjusted)
        comparisons[original_index]["mcnemar_p_holm"] = running


def analyze_records(
    records_path: Path | list[Path],
    *,
    experiment_path: Path,
    sample_path: Path,
    output_directory: Path,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    publication_files = ("aggregate-metrics.json", "paired-comparisons.json", "run-manifest.json", "category-results.csv", "limitations.md", "report.md")
    existing = [name for name in publication_files if (output_directory / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite publication files: {', '.join(existing)}")
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    sample = load_sample(sample_path)
    expected_ids = [str(item_id) for item_id in sample["item_ids"]]
    requested_models = [model["model_alias"] for model in experiment["models"]]
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    malformed = 0
    records_paths = [records_path] if isinstance(records_path, Path) else records_path
    for source_path in records_paths:
        for line in source_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            alias = record.get("model", {}).get("model_alias")
            item_id = record.get("item_id")
            if alias in requested_models and item_id in expected_ids:
                key = (alias, item_id)
                if record.get("status") == "completed" or key not in latest:
                    latest[key] = record

    results: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    coverage = {}
    model_metadata = {}
    for alias in requested_models:
        completed_ids = {item_id for model_alias, item_id in latest if model_alias == alias and latest[(model_alias, item_id)].get("status") == "completed"}
        coverage[alias] = {"expected": len(expected_ids), "completed": len(completed_ids), "missing": len(set(expected_ids) - completed_ids), "extra": len(completed_ids - set(expected_ids))}
        records = [latest[(alias, item_id)] for item_id in expected_ids if (alias, item_id) in latest and latest[(alias, item_id)].get("status") == "completed"]
        if records:
            model_metadata[alias] = {"model": records[0]["model"], "environment": records[0]["environment"]}
        buckets: dict[tuple[str, str], list[bool]] = defaultdict(list)
        for record in records:
            score = record["output"]["score"]
            value = bool(score["correct"])
            buckets[("overall", "all")].append(value)
            buckets[("language", str(score["language"]))].append(value)
            buckets[("category", str(score["category"]))].append(value)
        for (dimension_type, dimension), values in buckets.items():
            results[alias][f"{dimension_type}:{dimension}"] = {"dimension_type": dimension_type, "dimension": dimension, **_metric(values)}

    complete = malformed == 0 and all(item["missing"] == 0 and item["extra"] == 0 for item in coverage.values())
    condition_signatures = set()
    for alias in requested_models:
        representative = next((latest[(alias, item_id)] for item_id in expected_ids if (alias, item_id) in latest and latest[(alias, item_id)].get("status") == "completed"), None)
        if representative:
            model = representative["model"]
            check = representative["check"]
            condition_signatures.add((model.get("backend"), model.get("precision"), model.get("quantization"), check.get("dataset_revision"), check.get("dataset_sha256"), check.get("prompt_version"), check.get("scoring_version")))
    matched_conditions = len(condition_signatures) == 1 and len(model_metadata) == len(requested_models)
    comparable = complete and matched_conditions
    comparisons = []
    if comparable:
        for index, (left, right) in enumerate(itertools.combinations(requested_models, 2)):
            left_values = [bool(latest[(left, item_id)]["output"]["score"]["correct"]) for item_id in expected_ids]
            right_values = [bool(latest[(right, item_id)]["output"]["score"]["correct"]) for item_id in expected_ids]
            left_only, right_only, p_value = _mcnemar_exact(left_values, right_values)
            comparisons.append({"model_a": left, "model_b": right, "accuracy_difference_a_minus_b": sum(left_values) / len(left_values) - sum(right_values) / len(right_values), "paired_bootstrap_ci95": _bootstrap_difference(left_values, right_values, seed=experiment["seed"] + index, iterations=bootstrap_iterations), "a_correct_b_wrong": left_only, "a_wrong_b_correct": right_only, "mcnemar_p_value": p_value})
        _holm(comparisons)

    evidence_class = sample["evidence_class"]
    reporting_status = "confirmatory" if evidence_class == "confirmatory" and comparable else "preliminary" if evidence_class == "pilot" and comparable else "technical_or_incomplete"
    publishable_outcome = reporting_status == "confirmatory"
    output_directory.mkdir(parents=True, exist_ok=True)
    aggregate = {"schema_version": "1.0", "experiment_id": experiment["experiment_id"], "evidence_class": evidence_class, "reporting_status": reporting_status, "publishable_outcome": publishable_outcome, "complete_matched_coverage": complete, "matched_inference_conditions": matched_conditions, "coverage": coverage, "metrics": dict(results)}
    paired = {"schema_version": "1.0", "experiment_id": experiment["experiment_id"], "method": {"interval": "paired nonparametric bootstrap", "iterations": bootstrap_iterations, "test": "exact McNemar", "multiplicity": "Holm"}, "comparisons": comparisons}
    run_manifest = {"schema_version": "1.0", "experiment_id": experiment["experiment_id"], "raw_records": [{"sha256": _sha256(path)} for path in records_paths], "sample": {key: value for key, value in sample.items() if key != "item_ids"}, "models": model_metadata, "malformed_records": malformed, "sanitized": True}
    (output_directory / "aggregate-metrics.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_directory / "paired-comparisons.json").write_text(json.dumps(paired, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_directory / "run-manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_directory / "category-results.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=["model", "dimension_type", "dimension", "n", "correct", "accuracy", "ci95_low", "ci95_high"])
        writer.writeheader()
        for model, model_results in results.items():
            for metric in model_results.values():
                writer.writerow({"model": model, "dimension_type": metric["dimension_type"], "dimension": metric["dimension"], "n": metric["n"], "correct": metric["correct"], "accuracy": metric["accuracy"], "ci95_low": metric["ci95_wilson"][0], "ci95_high": metric["ci95_wilson"][1]})
    limitations = ["# Limitations", "", f"- Evidence class: `{evidence_class}`.", "- SafetyBench measures multiple-choice safety knowledge, not refusal behaviour or deployment safety.", "- Model comparisons are valid only for the frozen matched item set and recorded inference conditions."]
    if not complete:
        limitations.append("- Coverage is incomplete; paired conclusions are withheld.")
    if not matched_conditions:
        limitations.append("- Inference conditions differ or are missing across models; paired conclusions are withheld.")
    if evidence_class != "confirmatory":
        limitations.append("- This is not a confirmatory outcome and must not be presented as final evidence.")
    (output_directory / "limitations.md").write_text("\n".join(limitations) + "\n", encoding="utf-8")
    report_lines = ["# Safety evaluation report", "", f"Status: **{reporting_status.replace('_', ' ')}**", "", f"Frozen sample: {sample['item_count']} items (`{sample['item_ids_sha256'][:12]}`)", "", "| Model | Completed | Missing | Accuracy | 95% CI |", "|---|---:|---:|---:|---:|"]
    for alias in requested_models:
        overall = results.get(alias, {}).get("overall:all")
        accuracy = f"{overall['accuracy']:.3f}" if overall else "—"
        interval = f"{overall['ci95_wilson'][0]:.3f}–{overall['ci95_wilson'][1]:.3f}" if overall else "—"
        report_lines.append(f"| {alias} | {coverage[alias]['completed']} | {coverage[alias]['missing']} | {accuracy} | {interval} |")
    report_lines.extend(["", "Paired comparisons are available in `paired-comparisons.json`.", "", "No prompts, item-level answers, or benchmark text are included in this publication snapshot."])
    (output_directory / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {"output_directory": str(output_directory), "reporting_status": reporting_status, "publishable_outcome": publishable_outcome, "complete_matched_coverage": complete, "models": len(model_metadata), "items": sample["item_count"]}
