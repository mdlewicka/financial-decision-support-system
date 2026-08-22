import csv
import json
from dataclasses import asdict
from pathlib import Path

from backend.src.rag.evaluation import BenchmarkResult, quality_score


def export_benchmark(
    result: BenchmarkResult,
    output_directory: Path,
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "benchmark.json"
    csv_path = output_directory / "benchmark.csv"
    json_path.write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rows = [asdict(item) for item in result.items]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def export_comparison(
    results: list[BenchmarkResult],
    output_path: Path,
) -> Path:
    if not results:
        raise ValueError("results must not be empty.")
    ranking = sorted(results, key=quality_score, reverse=True)
    payload = {
        "best_configuration": ranking[0].configuration_name,
        "ranking": [
            {
                "rank": rank,
                "configuration_name": result.configuration_name,
                "quality_score": quality_score(result),
                **asdict(result.summary),
            }
            for rank, result in enumerate(ranking, start=1)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
