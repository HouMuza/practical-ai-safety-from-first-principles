"""Command-line entry point for portable safety-evaluation workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .hardware import inspect_machine, load_profiles, recommend_profile
from .planning import resolve_experiment


PIPELINE_DIR = Path(__file__).resolve().parents[2]
CHAPTER_DIR = PIPELINE_DIR.parent
PROFILES_PATH = PIPELINE_DIR / "configs" / "execution-profiles.json"


def _json(payload: Any) -> None:
    print(json.dumps(payload, indent=2))


def _model_manifests() -> list[dict[str, Any]]:
    manifests = []
    for path in sorted(CHAPTER_DIR.glob("*/configs/models.json")):
        with path.open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        manifests.append(
            {
                "family_id": manifest["family_id"],
                "display_name": manifest["display_name"],
                "models": len(manifest["models"]),
                "path": str(path.relative_to(CHAPTER_DIR)),
            }
        )
    return manifests


def _checks() -> list[dict[str, Any]]:
    checks = []
    for path in sorted((PIPELINE_DIR / "configs" / "benchmarks").glob("*.json")):
        with path.open(encoding="utf-8") as check_file:
            payload = json.load(check_file)
        checks.append({"check_id": payload.get("check_id", path.stem), "display_name": payload.get("display_name", path.stem), "path": str(path.relative_to(PIPELINE_DIR))})
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="safety-eval", description="Run reproducible model safety evaluations.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    inspect = subcommands.add_parser("inspect-machine", help="Report portable hardware facts.")
    inspect.add_argument("--json", action="store_true", dest="as_json")
    profiles = subcommands.add_parser("profiles", help="List execution profiles or recommend one.")
    profiles.add_argument("action", choices=("list", "recommend"))
    subcommands.add_parser("models", help="List registered model families.")
    subcommands.add_parser("checks", help="List registered safety checks.")
    plan = subcommands.add_parser("plan", help="Resolve an experiment without loading model weights.")
    plan.add_argument("experiment", type=Path)
    plan.add_argument("--profile", default=None)
    plan.add_argument("--max-items", type=int, default=None)
    freeze = subcommands.add_parser("freeze-sample", help="Freeze item IDs before inspecting outcomes.")
    freeze.add_argument("experiment", type=Path)
    freeze.add_argument("--dataset", required=True, type=Path)
    freeze.add_argument("--max-items", required=True, type=int)
    freeze.add_argument("--evidence-class", required=True, choices=("blinded_smoke", "pilot", "confirmatory"))
    freeze.add_argument("--output", required=True, type=Path)
    run = subcommands.add_parser("run", help="Execute one explicit model/check pair.")
    run.add_argument("experiment", type=Path)
    run.add_argument("--model", required=True, help="Model alias from the experiment manifest.")
    run.add_argument("--check", default=None, help="Check ID; optional when the experiment has one check.")
    run.add_argument("--dataset", required=True, type=Path, help="Local JSON or JSONL check data.")
    run.add_argument("--profile", default=None)
    run.add_argument("--max-items", type=int, default=None)
    run.add_argument("--allow-download", action="store_true", help="Allow model/tokenizer downloads when absent locally.")
    run.add_argument("--output", type=Path, default=None, help="Override the append-only JSONL path.")
    run.add_argument("--sample-manifest", type=Path, default=None, help="Use an exact frozen item-ID set.")
    analyze = subcommands.add_parser("analyze", help="Create a sanitized aggregate publication snapshot.")
    analyze.add_argument("--records", required=True, type=Path, action="append", help="JSONL source; repeat for runs from multiple machines.")
    analyze.add_argument("--experiment", required=True, type=Path)
    analyze.add_argument("--sample-manifest", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect-machine":
        machine = inspect_machine()
        if args.as_json:
            _json(machine.to_dict())
        else:
            memory = "unknown" if machine.memory_bytes is None else f"{machine.memory_bytes / (1024**3):.1f} GiB"
            accelerators = ", ".join(item.name for item in machine.accelerators) or "none detected"
            print(f"OS: {machine.operating_system} ({machine.architecture})")
            print(f"CPU threads: {machine.cpu_count}")
            print(f"Memory: {memory}")
            print(f"Accelerators: {accelerators}")
        return 0
    if args.command == "profiles":
        profiles = load_profiles(PROFILES_PATH)
        _json(profiles if args.action == "list" else recommend_profile(inspect_machine(), profiles))
        return 0
    if args.command == "models":
        _json(_model_manifests())
        return 0
    if args.command == "checks":
        _json(_checks())
        return 0
    if args.command == "plan":
        if args.max_items is not None and args.max_items < 1:
            raise SystemExit("--max-items must be at least 1")
        _json(resolve_experiment(args.experiment, profile_override=args.profile, max_items_override=args.max_items))
        return 0
    if args.command == "freeze-sample":
        if args.max_items < 1:
            raise SystemExit("--max-items must be at least 1")
        plan = resolve_experiment(args.experiment, max_items_override=args.max_items)
        if len(plan["checks"]) != 1 or plan["checks"][0]["check_id"] != "safetybench":
            raise SystemExit("The built-in freezer currently supports one SafetyBench check")
        from .adapters.safetybench import SafetyBenchCheck
        from .sampling import freeze_sample

        check = SafetyBenchCheck(plan["checks"][0], args.dataset)
        manifest = freeze_sample(check, experiment_id=plan["experiment"]["experiment_id"], seed=plan["experiment"]["seed"], max_items=args.max_items, evidence_class=args.evidence_class, output_path=args.output)
        _json({key: value for key, value in manifest.items() if key != "item_ids"})
        return 0
    if args.command == "run":
        if args.max_items is not None and args.max_items < 1:
            raise SystemExit("--max-items must be at least 1")
        if args.sample_manifest is not None and args.max_items is not None:
            raise SystemExit("Use the frozen sample size; do not combine --sample-manifest with --max-items")
        plan = resolve_experiment(args.experiment, profile_override=args.profile, max_items_override=args.max_items)
        frozen_item_ids = None
        if args.sample_manifest is not None:
            from .sampling import attach_sample_to_plan, load_sample

            sample = load_sample(args.sample_manifest)
            plan = attach_sample_to_plan(plan, sample)
            frozen_item_ids = [str(item_id) for item_id in sample["item_ids"]]
        try:
            model_spec = next(model for model in plan["models"] if model["alias"] == args.model)
        except StopIteration as error:
            aliases = ", ".join(model["alias"] for model in plan["models"])
            raise SystemExit(f"Model {args.model!r} is not in the experiment; choose: {aliases}") from error
        if args.check is None:
            if len(plan["checks"]) != 1:
                raise SystemExit("--check is required when an experiment contains multiple checks")
            check_manifest = plan["checks"][0]
        else:
            try:
                check_manifest = next(check for check in plan["checks"] if check["check_id"] == args.check)
            except StopIteration as error:
                raise SystemExit(f"Check {args.check!r} is not in the experiment") from error

        from .adapters.safetybench import SafetyBenchCheck
        from .adapters.transformers import TransformersAdapter
        from .runner import run_check

        if check_manifest["check_id"] != "safetybench":
            raise SystemExit(f"No built-in runtime adapter is registered for {check_manifest['check_id']!r}")
        check = SafetyBenchCheck(check_manifest, args.dataset, frozen_item_ids=frozen_item_ids)
        model = TransformersAdapter(
            model_spec,
            precision=plan["profile"]["precision"],
            quantization=plan["profile"]["quantization"],
            allow_download=args.allow_download,
        )
        records_path = args.output or Path(plan["output_directory"]) / plan["run_id"] / f"{check_manifest['check_id']}.jsonl"
        counts = run_check(plan, model, check, records_path=records_path)
        _json({"run_id": plan["run_id"], "records_path": str(records_path), **counts})
        return 0
    if args.command == "analyze":
        if args.bootstrap_iterations < 100:
            raise SystemExit("--bootstrap-iterations must be at least 100")
        from .analysis import analyze_records

        _json(analyze_records(args.records, experiment_path=args.experiment, sample_path=args.sample_manifest, output_directory=args.output, bootstrap_iterations=args.bootstrap_iterations))
        return 0
    return 2
