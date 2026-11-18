"""Control for runtime effects by timing FP32 and INT8 with the same ONNX Runtime."""
import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import onnxruntime as ort
import torch
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from bert_model import BertModerator
from bert_onnx import BertOnnxModerator
from export_bert_onnx import export_fp32_graph
from train_classical import cpu_name


def measure(folder, rows, timed_indices):
    model = BertOnnxModerator(folder, threads=1)
    for i in timed_indices[:10]:
        model.predict_one(rows[i]["text"])
    start = time.perf_counter()
    predicted = [model.predict_one(rows[i]["text"])[0] for i in timed_indices]
    elapsed = time.perf_counter() - start
    return {
        "mean_latency_ms": 1000 * elapsed / len(timed_indices),
        "timed_subset_macro_f1": f1_score([int(rows[i]["label"]) for i in timed_indices],
                                          predicted, average="macro"),
    }


def main(args):
    with args.validation.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indices = list(range(len(rows)))
    random.Random(42).shuffle(indices)
    indices = indices[:200]
    if len(indices) < 200:
        raise ValueError("The paired validation control requires at least 200 examples.")
    report = {
        "validation_sha256": hashlib.sha256(args.validation.read_bytes()).hexdigest(),
        "rows": len(rows), "runs": len(indices), "warmups": 10, "shuffle_seed": 42, "batch_size": 1,
        "intra_op_threads": 1, "inter_op_threads": 1, "execution_provider": "CPUExecutionProvider",
        "cpu": cpu_name(), "onnxruntime_version": ort.__version__,
    }
    fp_folder = args.artifacts / "bert_fp32"
    int_folder = args.artifacts / "bert_int8"
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=1):
        temp = Path(tmp)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "moderation.json"):
            shutil.copyfile(fp_folder / name, temp / name)
        export_fp32_graph(BertModerator(fp_folder), temp / "model.onnx")
        report["fp32_onnx"] = measure(temp, rows, indices)
        report["int8_onnx"] = measure(int_folder, rows, indices)
        model = BertOnnxModerator(int_folder, threads=1)
        predictions = model.predict_many([row["text"] for row in rows], batch_size=16)
        report["int8_validation_macro_f1"] = f1_score([int(row["label"]) for row in rows],
                                                      predictions, average="macro")
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=Path("data/processed/validation.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("results/ort_paired_validation.json"))
    main(parser.parse_args())
