"""Automated experiment suite runner."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.config import BASE_DIR


MEAN_NDCG_RE = re.compile(r"Mean NDCG@10:\s*([0-9]*\.?[0-9]+)")
MEAN_RECALL_RE = re.compile(r"Mean Recall@10:\s*([0-9]*\.?[0-9]+)")
MEDIAN_NDCG_RE = re.compile(r"Median NDCG@10:\s*([0-9]*\.?[0-9]+)")
MEDIAN_RECALL_RE = re.compile(r"Median Recall@10:\s*([0-9]*\.?[0-9]+)")


def _parse_metrics(stdout: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    mean_ndcg = MEAN_NDCG_RE.search(stdout)
    mean_recall = MEAN_RECALL_RE.search(stdout)
    median_ndcg = MEDIAN_NDCG_RE.search(stdout)
    median_recall = MEDIAN_RECALL_RE.search(stdout)

    if mean_ndcg:
        metrics["mean_ndcg_at_10"] = float(mean_ndcg.group(1))
    if mean_recall:
        metrics["mean_recall_at_10"] = float(mean_recall.group(1))
    if median_ndcg:
        metrics["median_ndcg_at_10"] = float(median_ndcg.group(1))
    if median_recall:
        metrics["median_recall_at_10"] = float(median_recall.group(1))

    return metrics


def _run_command(command: List[str], env: Dict[str, str]) -> Tuple[int, str, str]:
    completed = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
    )
    return completed.returncode, completed.stdout, completed.stderr


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_suite(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError("Experiment suite config must be a JSON object")
    if "experiments" not in data or not isinstance(data["experiments"], list):
        raise ValueError("Experiment suite config must include an 'experiments' list")
    return data


def run_suite(config_path: Path, output_root: Path) -> Path:
    suite = _load_suite(config_path)

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    suite_name = config_path.stem
    run_dir = output_root / f"{suite_name}_{timestamp}"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    for key, value in suite.get("base_env", {}).items():
        base_env[str(key)] = str(value)

    default_commands = suite.get("commands", ["evaluate"])
    if not isinstance(default_commands, list) or not default_commands:
        default_commands = ["evaluate"]

    results: list[dict[str, Any]] = []

    for exp in suite["experiments"]:
        if not isinstance(exp, dict):
            continue

        name = str(exp.get("name", "unnamed_experiment"))
        exp_env = base_env.copy()
        for key, value in exp.get("env", {}).items():
            exp_env[str(key)] = str(value)

        commands = exp.get("commands", default_commands)
        if not isinstance(commands, list) or not commands:
            commands = ["evaluate"]

        experiment_result: dict[str, Any] = {
            "name": name,
            "env": exp.get("env", {}),
            "commands": commands,
            "status": "success",
            "metrics": {},
            "command_results": [],
        }

        for command_name in commands:
            module = None
            if command_name == "evaluate":
                module = "src.cli.evaluate"
            elif command_name == "generate_submission":
                module = "src.cli.generate_submission"
            else:
                experiment_result["status"] = "failed"
                experiment_result["command_results"].append(
                    {
                        "command": command_name,
                        "return_code": -1,
                        "error": f"Unsupported command: {command_name}",
                    }
                )
                continue

            cmd = [sys.executable, "-m", module]
            code, stdout, stderr = _run_command(cmd, exp_env)

            log_prefix = logs_dir / f"{name}_{command_name}"
            _write_text(log_prefix.with_suffix(".stdout.log"), stdout)
            _write_text(log_prefix.with_suffix(".stderr.log"), stderr)

            cmd_result = {
                "command": command_name,
                "return_code": code,
                "stdout_log": str(log_prefix.with_suffix(".stdout.log").relative_to(BASE_DIR)),
                "stderr_log": str(log_prefix.with_suffix(".stderr.log").relative_to(BASE_DIR)),
            }
            experiment_result["command_results"].append(cmd_result)

            if command_name == "evaluate":
                metrics = _parse_metrics(stdout)
                experiment_result["metrics"].update(metrics)

            if code != 0:
                experiment_result["status"] = "failed"

        results.append(experiment_result)

    results_json_path = run_dir / "results.json"
    results_csv_path = run_dir / "results.csv"
    summary_md_path = run_dir / "summary.md"

    with results_json_path.open("w", encoding="utf-8") as file_obj:
        json.dump({"suite": suite_name, "config": str(config_path), "results": results}, file_obj, indent=2)

    with results_csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "name",
                "status",
                "mean_ndcg_at_10",
                "mean_recall_at_10",
                "median_ndcg_at_10",
                "median_recall_at_10",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "name": item.get("name"),
                    "status": item.get("status"),
                    "mean_ndcg_at_10": item.get("metrics", {}).get("mean_ndcg_at_10", ""),
                    "mean_recall_at_10": item.get("metrics", {}).get("mean_recall_at_10", ""),
                    "median_ndcg_at_10": item.get("metrics", {}).get("median_ndcg_at_10", ""),
                    "median_recall_at_10": item.get("metrics", {}).get("median_recall_at_10", ""),
                }
            )

    summary_lines = [
        f"# Experiment Suite: {suite_name}",
        "",
        f"Config: `{config_path}`",
        f"Run directory: `{run_dir}`",
        "",
        "| Experiment | Status | Mean NDCG@10 | Mean Recall@10 |",
        "|---|---|---:|---:|",
    ]

    for item in results:
        m = item.get("metrics", {})
        summary_lines.append(
            f"| {item.get('name')} | {item.get('status')} | {m.get('mean_ndcg_at_10', '')} | {m.get('mean_recall_at_10', '')} |"
        )

    _write_text(summary_md_path, "\n".join(summary_lines) + "\n")

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated recommendation experiments")
    parser.add_argument(
        "--config",
        default="src/cli/suites/default.json",
        help="Path to experiment suite config JSON",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/experiments",
        help="Directory where experiment run artifacts are saved",
    )

    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = BASE_DIR / output_root

    run_dir = run_suite(config_path=config_path, output_root=output_root)
    print(f"Experiment suite finished. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
