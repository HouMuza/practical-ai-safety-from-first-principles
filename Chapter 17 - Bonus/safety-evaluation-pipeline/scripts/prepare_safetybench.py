"""Merge pinned official SafetyBench questions and answers into local JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_head(source: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def prepare(source: Path, output: Path, revision: str, languages: list[str]) -> dict:
    source = source.resolve()
    actual_revision = repository_head(source)
    if actual_revision != revision:
        raise ValueError(f"Expected SafetyBench revision {revision}, found {actual_revision}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    source_files = []
    counts: Counter[str] = Counter()
    total = 0

    with temporary.open("w", encoding="utf-8") as destination:
        for language in languages:
            question_path = source / "opensource_data" / f"test_{language}.json"
            answer_path = source / "opensource_data" / f"test_answers_{language}.json"
            questions = json.loads(question_path.read_text(encoding="utf-8"))
            answers = json.loads(answer_path.read_text(encoding="utf-8"))
            source_files.extend(
                [
                    {"path": str(question_path.relative_to(source)), "sha256": sha256(question_path)},
                    {"path": str(answer_path.relative_to(source)), "sha256": sha256(answer_path)},
                ]
            )
            prompt_language = "zh" if language.startswith("zh") else "en"
            for question in questions:
                source_id = str(question["id"])
                if source_id not in answers:
                    raise ValueError(f"No answer for {language} item {source_id}")
                answer = answers[source_id]
                category = str(question["category"])
                if answer.get("category") != category:
                    raise ValueError(f"Category mismatch for {language} item {source_id}")
                item = {
                    "item_id": f"{language}:{source_id}",
                    "source_id": source_id,
                    "question": question["question"],
                    "options": question["options"],
                    "answer": answer["answer"],
                    "language": prompt_language,
                    "dataset_split": language,
                    "category": category,
                }
                destination.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                counts[f"{language}/{category}"] += 1
                total += 1
    temporary.replace(output)
    metadata = {
        "schema_version": "1.0",
        "dataset": "thu-coai/SafetyBench",
        "source_revision": revision,
        "languages": languages,
        "items": total,
        "counts_by_split_and_category": dict(sorted(counts.items())),
        "source_files": source_files,
        "output": {"path": output.name, "sha256": sha256(output)},
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--languages", nargs="+", choices=("en", "zh", "zh_subset"), default=["en", "zh"])
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.revision, args.languages), indent=2))


if __name__ == "__main__":
    main()
