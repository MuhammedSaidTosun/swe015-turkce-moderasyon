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

from bert_model import BertModerator, quantize_for_cpu
from train_classical import cpu_name


def load_torch_int8(folder):
    model = BertModerator(folder)
    if model.metadata["precision"] == "FP32":
        model.model = quantize_for_cpu(model.model)
    elif model.metadata["precision"] != "INT8":
        raise ValueError(f"Expected FP32 or INT8 PyTorch weights in {folder}.")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path,
                        help="PyTorch INT8 weights directory or FP32 weights directory to quantize. "
                             "Defaults to historical artifacts/bert_int8/weights.pt when present, "
                             "otherwise uses the released artifacts/bert_fp32 model.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    source = args.model_dir
    if source is None:
        historical = Path("artifacts/bert_int8")
        source = historical if (historical / "weights.pt").is_file() else Path("artifacts/bert_fp32")
    if not (source / "weights.pt").is_file():
        parser.error(f"Missing {source / 'weights.pt'}. Extract the model release or supply --model-dir.")
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
            predictor = load_torch_int8(source)
            source_precision = predictor.metadata["precision"]
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
        "source_precision": source_precision, "pytorch_quantized_engine": torch.backends.quantized.engine,
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
