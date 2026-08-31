"""Small dependency-free local API for the safety-evaluation platform."""

from __future__ import annotations

import json
import os
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


CHAPTER_DIR = Path(__file__).resolve().parents[3]
def load_models(chapter_dir: Path = CHAPTER_DIR) -> dict:
    families = []
    for path in sorted(chapter_dir.glob("*/configs/models.json")):
        with path.open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        families.append(manifest)
    return {"families": families}


def discover_runs(chapter_dir: Path = CHAPTER_DIR) -> dict:
    runs = []
    pattern = "*/experiments/*/runs/*/*.jsonl"
    for path in sorted(chapter_dir.glob(pattern)):
        records = []
        malformed = 0
        with path.open(encoding="utf-8") as raw_records:
            for line in raw_records:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed += 1
        if not records:
            continue
        completed = [record for record in records if record.get("status") == "completed"]
        failed = sum(record.get("status") == "failed" for record in records)
        languages: Counter[str] = Counter()
        categories: Counter[str] = Counter()
        correct = 0
        scored = 0
        for record in completed:
            score = (record.get("output") or {}).get("score") or {}
            if score.get("language"):
                languages[str(score["language"])] += 1
            if score.get("category"):
                categories[str(score["category"])] += 1
            if isinstance(score.get("correct"), bool):
                scored += 1
                correct += int(score["correct"])
        first = records[0]
        model = first.get("model", {})
        check = first.get("check", {})
        evidence_class = "technical_smoke" if len(completed) <= 20 else "sampled"
        publishable_outcome = evidence_class != "technical_smoke"
        runs.append(
            {
                "run_id": first.get("run_id"),
                "experiment_id": first.get("experiment_id"),
                "model": {key: model.get(key) for key in ("family_id", "model_alias", "revision", "backend", "device", "precision", "quantization")},
                "check_id": check.get("check_id"),
                "dataset_revision": check.get("dataset_revision"),
                "dataset_sha256": check.get("dataset_sha256"),
                "completed": len(completed),
                "failed": failed,
                "malformed": malformed,
                "languages": dict(sorted(languages.items())),
                "categories": dict(sorted(categories.items())),
                "accuracy": correct / scored if scored and publishable_outcome else None,
                "evidence_class": evidence_class,
                "publishable_outcome": publishable_outcome,
                "completed_at": max((record.get("completed_at") or "" for record in records), default=""),
            }
        )
    return {
        "runs": runs,
        "totals": {
            "runs": len(runs),
            "completed_records": sum(run["completed"] for run in runs),
            "failed_records": sum(run["failed"] for run in runs),
            "publishable_outcomes": sum(run["publishable_outcome"] for run in runs),
        },
    }


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "SafetyEvaluationAPI/0.1"

    def send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        origin = self.headers.get("Origin")
        if origin in {"http://localhost:3000", "http://127.0.0.1:3000"}:
            self.send_header("Access-Control-Allow-Origin", origin)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/api/health":
            self.send_json(200, {"status": "ok", "service": "safety-evaluation-api"})
            return
        if path == "/api/models":
            self.send_json(200, load_models())
            return
        if path == "/api/experiments":
            self.send_json(200, discover_runs())
            return
        if path == "/api/summary":
            payload = discover_runs()
            payload["models"] = load_models()["families"]
            self.send_json(200, payload)
            return
        self.send_json(404, {"error": "not_found", "path": path})

    def log_message(self, message: str, *args: object) -> None:
        print(f"[backend] {self.address_string()} - {message % args}")


def main() -> None:
    host = os.environ.get("SAFETY_API_HOST", "127.0.0.1")
    port = int(os.environ.get("SAFETY_API_PORT", "8788"))
    server = ThreadingHTTPServer((host, port), ApiHandler)
    print(f"Safety evaluation API: http://{host}:{port}")
    print(f"Health check: http://{host}:{port}/api/health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
