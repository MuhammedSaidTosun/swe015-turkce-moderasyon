"""Evaluate all three deployed models on the same held-out messages and CPU."""

import argparse
import csv
import hashlib
import json
import platform
import random
import time
from importlib import metadata
from pathlib import Path

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from classical import load_model
from neural import NeuralModerator
from train_classical import cpu_name, macro_f1, read_split


def measure(model, texts, labels, timing_texts, artifact):
    predictions = [model.predict_one(text)[0] for text in texts]
    for text in timing_texts[:10]:
        model.predict_one(text)
    elapsed_ns = 0
    for text in timing_texts:
        start = time.perf_counter_ns()
        model.predict_one(text)
        elapsed_ns += time.perf_counter_ns() - start

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel().tolist()
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1, zero_division=0
    )
    return {
        "macro_f1": macro_f1(labels, predictions),
        "size_mb": artifact.stat().st_size / 1_000_000,
        "mean_cpu_latency_ms": elapsed_ns / len(timing_texts) / 1_000_000,
        "runs": len(timing_texts),
        "offensive_precision": float(precision),
        "offensive_recall": float(recall),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()
    if args.runs < 100:
        parser.error("--runs must be at least 100")

    test_file = args.data_dir / "test.csv"
    split_digest = hashlib.sha256(test_file.read_bytes()).hexdigest()
    split_report = json.loads((args.data_dir / "split_summary.json").read_text(encoding="utf-8"))
    if split_report["splits"]["test"]["sha256"] != split_digest:
        raise ValueError("Test CSV differs from the data preparation manifest.")

    texts, labels = read_split(test_file)
    indices = list(range(len(texts)))
    random.Random(42).shuffle(indices)
    timing_texts = [texts[indices[i % len(indices)]] for i in range(args.runs)]
    cpu = cpu_name()
    artifacts = {"rules": "rules.json", "tfidf": "tfidf.joblib", "neural": "neural.npz"}
    rows = []
    hashes = {}
    with threadpool_limits(limits=1):
        for name, filename in artifacts.items():
            path = args.artifact_dir / filename
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
            model = NeuralModerator(path) if name == "neural" else load_model(name, args.artifact_dir)
            metrics = measure(model, texts, labels, timing_texts, path)
            rows.append({"model": name, **metrics, "cpu": cpu})

    report = {
        "evaluation_split": "test",
        "evaluation_sha256": split_digest,
        "test_counts": {"total": len(texts), "normal": labels.count(0), "offensive": labels.count(1)},
        "artifact_sha256": hashes,
        "hardware": {"cpu": cpu, "platform": platform.platform(), "machine": platform.machine()},
        "runtime": {
            "python": platform.python_version(),
            **{name: metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        },
        "latency_protocol": "Batch size 1, one numerical-library thread, predict_one includes preprocessing. "
                            "Seed 42, same shuffled test messages for each model, 10 warm-ups, wall-clock mean "
                            "with models already loaded. HTTP and UI time excluded.",
        "size_definition": "Deployed artifact bytes / 1,000,000 excluding runtime and shared code.",
        "models": rows,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "neural_test.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.results_dir / "comparison_test.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
