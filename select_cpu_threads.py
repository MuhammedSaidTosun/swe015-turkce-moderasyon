"""Measure the original PyTorch INT8 CPU thread counts using validation texts only."""

import argparse
import csv
import hashlib
import json
import os
import random
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import torch
from threadpoolctl import threadpool_limits

from bert_model import BertModerator
from train_classical import cpu_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/bert_int8"),
                        help="PyTorch INT8 model directory containing weights.pt")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    validation = args.data_dir / "validation.csv"
    with validation.open(encoding="utf-8", newline="") as source:
        messages = [row["text"] for row in csv.DictReader(source)]
    warmups, runs = 10, 120
    sampled = random.Random(42).sample(messages, warmups + runs)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_interop_threads(1)
    measurements = []
    for threads in (1, 2, 4):
        torch.set_num_threads(threads)
        with threadpool_limits(limits=threads):
            predictor = BertModerator(args.model_dir)
            for text in sampled[:warmups]:
                predictor.predict_one(text)
            durations = []
            for text in sampled[warmups:]:
                start = perf_counter_ns()
                predictor.predict_one(text)
                durations.append((perf_counter_ns() - start) / 1_000_000)
            del predictor
        durations.sort()
        row = {"threads": threads, "mean_ms": sum(durations) / runs,
               "p95_ms": durations[ceil(0.95 * runs) - 1], "max_ms": durations[-1]}
        measurements.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "split": "validation", "sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
        "cpu": cpu_name(), "seed": 42, "warmups": warmups, "runs": runs,
        "measurements": measurements,
        "selected_threads": min(measurements, key=lambda row: row["mean_ms"])["threads"],
        "selection": "Lowest mean single-message latency among 1, 2 and 4 CPU threads.",
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "cpu_thread_validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
