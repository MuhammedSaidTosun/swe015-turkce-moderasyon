"""Evaluate all three deployed models on the same held-out messages and CPU."""

import argparse
import csv
import hashlib
import json
import platform
import random
import time
from importlib import metadata
from math import ceil
from pathlib import Path

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from classical import load_model
from neural import NeuralModerator
from train_classical import cpu_name, macro_f1, read_split


def measure(model, texts, labels, timing_texts, artifact, predictions=None, latency_budget_ms=None):
    if predictions is None:
        predictions = [model.predict_one(text)[0] for text in texts]
    for text in timing_texts[:10]:
        model.predict_one(text)
    durations_ms = []
    for text in timing_texts:
        start = time.perf_counter_ns()
        model.predict_one(text)
        durations_ms.append((time.perf_counter_ns() - start) / 1_000_000)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel().tolist()
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1, zero_division=0
    )
    metrics = {
        "macro_f1": macro_f1(labels, predictions),
        "size_mb": sum(p.stat().st_size for p in deployment_files(artifact)) / 1_000_000,
        "mean_cpu_latency_ms": sum(durations_ms) / len(durations_ms),
        "runs": len(timing_texts),
        "offensive_precision": float(precision),
        "offensive_recall": float(recall),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    if latency_budget_ms is not None:
        ordered = sorted(durations_ms)
        metrics.update(
            p95_cpu_latency_ms=ordered[ceil(0.95 * len(ordered)) - 1],
            max_cpu_latency_ms=ordered[-1],
            over_budget_count=sum(ms > latency_budget_ms for ms in ordered),
        )
    return metrics


def deployment_files(path):
    if not path.is_dir():
        return [path]
    return sorted(file for file in path.rglob("*") if file.is_file())


def main(include_transformers=False):
    description = ("Evaluate all five CPU models on the same test split."
                   if include_transformers else __doc__)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--runs", type=int, default=200)
    if include_transformers:
        parser.add_argument("--threads", type=int, help="CPU threads for all five models")
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
    latency_budget_ms = 50.0 if include_transformers else None
    threads = 1
    if include_transformers:
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from bert_model import load_deployed_bert, CPU_THREADS

        threads = CPU_THREADS if args.threads is None else args.threads
        if threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        artifacts.update(bert_fp32="bert_fp32", bert_int8="bert_int8")
        # Check both exports before beginning the five-model comparison.
        for name in ("bert_fp32", "bert_int8"):
            folder = args.artifact_dir / name
            if not (folder / "moderation.json").is_file() or not any(
                (folder / file).is_file() for file in ("model.onnx", "weights.pt")
            ):
                parser.error(f"Missing deployed weights or configuration in {folder}")
    rows = []
    hashes = {}
    with threadpool_limits(limits=threads):
        for name, filename in artifacts.items():
            path = args.artifact_dir / filename
            for file in deployment_files(path):
                with file.open("rb") as stream:
                    hashes[file.relative_to(args.artifact_dir).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
            if name.startswith("bert_"):
                model = load_deployed_bert(path, threads=threads)
            elif name == "neural":
                model = NeuralModerator(path)
            else:
                model = load_model(name, args.artifact_dir)
            # Dynamic INT8 activation scales can vary with batch composition.
            # Compute INT8 labels with the same one-message call used by the chat.
            predictions = model.predict_many(texts, batch_size=16) if name == "bert_fp32" else None
            metrics = measure(model, texts, labels, timing_texts, path, predictions, latency_budget_ms)
            if include_transformers:
                metrics["accuracy"] = (metrics["tp"] + metrics["tn"]) / len(labels)
                print(f"{name}: macro-F1 {metrics['macro_f1']:.4f}, "
                      f"mean {metrics['mean_cpu_latency_ms']:.3f} ms", flush=True)
            rows.append({"model": name, **metrics, "cpu": cpu})
            del model

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
        "latency_protocol": f"Batch size 1, numerical-library threads={threads}, predict_one includes preprocessing. "
                            "Seed 42, same shuffled test messages for each model, 10 warm-ups, wall-clock mean "
                            "with models already loaded. HTTP and UI time excluded.",
        "size_definition": "Deployed artifact bytes / 1,000,000 excluding runtime and shared code.",
        "models": rows,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_name = "neural_test.json"
    csv_name = "comparison_test.csv"
    if include_transformers:
        summary_name = "five_model_test.json"
        csv_name = "five_model_test.csv"
        use_onnx = (args.artifact_dir / "bert_int8" / "model.onnx").is_file()
        report["runtime"].update(
            torch=metadata.version("torch"), transformers=metadata.version("transformers"),
            fp32_backend="PyTorch CPU", torch_threads=torch.get_num_threads(),
            int8_backend="ONNX Runtime CPUExecutionProvider" if use_onnx else str(torch.backends.quantized.engine),
            tokenizer_parallelism=False,
        )
        if use_onnx:
            report["runtime"]["onnxruntime"] = metadata.version("onnxruntime")
            report["runtime"]["onnx_intra_op_threads"] = threads
        int8 = next(row for row in rows if row["model"] == "bert_int8")
        report["cpu_budget"] = {
            "target_mean_ms": latency_budget_ms,
            "int8_measured_mean_ms": int8["mean_cpu_latency_ms"],
            "met": int8["mean_cpu_latency_ms"] <= latency_budget_ms,
        }
        report["size_definition"] = ("Each model's independently deployable files in decimal MB. "
                                     "Transformer directories include weights, tokenizer and configuration.")
    (args.results_dir / summary_name).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.results_dir / csv_name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
