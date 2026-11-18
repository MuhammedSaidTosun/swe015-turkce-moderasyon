"""Train and benchmark both classical SWE015 moderation baselines on CPU."""

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from classical import RuleModerator, TfidfModerator, load_model, normalize_text


def read_split(path):
    texts, labels = [], []
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain text and label columns")
        for row_num, row in enumerate(reader, start=2):
            if not row["text"].strip() or row["label"] not in {"0", "1"}:
                raise ValueError(f"Invalid text or label at {path}:{row_num}")
            texts.append(row["text"])
            labels.append(int(row["label"]))
    if set(labels) != {0, 1}:
        raise ValueError(f"{path} must contain both classes, labeled 0 and 1")
    return texts, labels


def macro_f1(actual, predicted):
    return float(f1_score(actual, predicted, labels=[0, 1], average="macro", zero_division=0))


def fit_tfidf(train, validation):
    train_text, train_labels = train
    val_text, val_labels = validation
    vectorizer = TfidfVectorizer(
        preprocessor=normalize_text, lowercase=False, token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2), min_df=2 if len(train_text) >= 100 else 1,
        max_features=70000, sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(train_text)
    val_features = vectorizer.transform(val_text)
    candidates = []
    for c in [0.5, 2.0]:
        model = LogisticRegression(C=c, class_weight="balanced", solver="liblinear",
                                   random_state=42, max_iter=300)
        model.fit(train_features, train_labels)
        candidates.append((macro_f1(val_labels, model.predict(val_features)), model))
    best_score, best_model = max(candidates, key=lambda item: item[0])
    tuning = {"selected_C": best_model.C, "validation_macro_f1": best_score,
              "C_candidates": [0.5, 2.0]}
    return TfidfModerator(vectorizer, best_model), tuning


def cpu_name():
    if platform.system() == "Darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        info = subprocess.run(["system_profiler", "SPHardwareDataType"],
                              capture_output=True, text=True, check=False)
        for line in info.stdout.splitlines():
            if line.strip().startswith("Chip:"):
                return line.split(":", 1)[-1].strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[-1].strip()
    return platform.processor() or platform.machine()


def benchmark(model, texts, labels, benchmark_texts, model_file):
    if isinstance(model, TfidfModerator):
        predictions = model.predict_many(texts)
    else:
        predictions = [model.predict_one(text)[0] for text in texts]
    for text in benchmark_texts[:10]:
        model.predict_one(text)
    elapsed = []
    for text in benchmark_texts:
        start = time.perf_counter_ns()
        model.predict_one(text)
        elapsed.append(time.perf_counter_ns() - start)
    return {
        "macro_f1": macro_f1(labels, predictions),
        "size_mb": model_file.stat().st_size / 1_000_000,
        "mean_cpu_latency_ms": sum(elapsed) / len(elapsed) / 1_000_000,
        "runs": len(elapsed),
    }


def run(args):
    train = read_split(args.data_dir / "train.csv")
    validation = read_split(args.data_dir / "validation.csv")
    evaluation_path = args.data_dir / f"{args.eval_split}.csv"
    texts, labels = validation if args.eval_split == "validation" else read_split(evaluation_path)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    tfidf, tuning = fit_tfidf(train, validation)
    paths = {"rules": args.artifact_dir / "rules.json", "tfidf": args.artifact_dir / "tfidf.joblib"}
    RuleModerator().save(paths["rules"])
    joblib.dump(tfidf, paths["tfidf"], compress=3)

    indices = list(range(len(texts)))
    random.Random(42).shuffle(indices)
    benchmark_texts = [texts[indices[i % len(indices)]] for i in range(args.runs)]
    hardware = cpu_name()
    rows = []
    for name, path in paths.items():
        model = load_model(name, args.artifact_dir)
        metrics = benchmark(model, texts, labels, benchmark_texts, path)
        rows.append({"model": name, **metrics, "cpu": hardware})
    report = {
        "evaluation_split": args.eval_split,
        "evaluation_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        "rows_by_split": {"train": len(train[0]), "validation": len(validation[0]),
                          args.eval_split: len(texts)},
        "tfidf_selection": tuning,
        "latency_protocol": "Loaded model, batch=1, includes normalization and prediction; 10 warmups; "
                            "identical seeded shuffle of held-out messages; one numerical library thread.",
        "size_definition": "Deployed artifact on disk in decimal MB. Rules JSON and compressed TF-IDF "
                           "joblib exclude shared code and runtime dependencies.",
        "hardware": {"cpu": hardware, "platform": platform.platform(), "machine": platform.machine()},
        "models": rows,
    }
    stem = f"classical_{args.eval_split}"
    (args.results_dir / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.results_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--eval-split", choices=["validation", "test"], default="test")
    parser.add_argument("--runs", type=int, default=200, help="Single-message CPU timings, at least 100")
    args = parser.parse_args()
    if args.runs < 100:
        parser.error("--runs must be at least 100")
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
