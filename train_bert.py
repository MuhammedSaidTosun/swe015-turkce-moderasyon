"""Fine-tune Turkish BERT and export two independently deployable CPU models."""

import argparse
import csv
import hashlib
import json
import math
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from bert_model import prepare_text, quantize_for_cpu
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [r["text"] for r in rows], np.array([int(r["label"]) for r in rows])


def encode(texts, tokenizer, max_length, turkish_lowercase):
    return tokenizer(
        [prepare_text(t, turkish_lowercase) for t in texts],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def iter_batches(encoded, labels, indices, tokenizer, batch_size, device):
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        features = {k: [encoded[k][j] for j in batch_indices] for k in encoded}
        yield (
            tokenizer.pad(features, padding=True, return_tensors="pt").to(device),
            torch.from_numpy(labels[batch_indices]).to(device),
        )


@torch.inference_mode()
def validation_probabilities(model, encoded, labels, tokenizer, batch_size, device):
    model.eval()
    output = []
    for features, _ in iter_batches(encoded, labels, np.arange(len(labels)), tokenizer, batch_size, device):
        output.extend(model(**features).logits.softmax(-1)[:, 1].cpu().tolist())
    return np.array(output)


def export(model, tokenizer, output, metadata):
    output.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(output)
    tokenizer.save_pretrained(output)
    torch.save(model.state_dict(), output / "weights.pt")
    (output / "moderation.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def install_exports(staging, destinations):
    """Keep the previous model pair available if either replacement fails."""
    backups, installed = [], []
    try:
        for dest in destinations:
            backup = staging / (dest.name + ".old")
            if dest.exists():
                dest.replace(backup)
                backups.append((dest, backup))
            (staging / dest.name).replace(dest)
            installed.append(dest)
    except OSError:
        for dest in reversed(installed):
            dest.replace(staging / dest.name)
        for dest, backup in reversed(backups):
            backup.replace(dest)
        raise


def main(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    model_loader = {"revision": args.revision} if args.revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, **model_loader)
    config = AutoConfig.from_pretrained(args.base_model, **model_loader)
    if config.model_type != "bert":
        raise ValueError(f"Expected a BERT encoder, received {config.model_type}.")
    total_layers = config.num_hidden_layers
    if args.layers:
        if not 1 <= args.layers <= total_layers:
            raise ValueError(f"--layers must be between 1 and {total_layers}")
        config.num_hidden_layers = args.layers
    config.architectures = ["BertForSequenceClassification"]
    config.num_labels = 2
    config.id2label = {0: "NOT", 1: "OFF"}
    config.label2id = {v: k for k, v in config.id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, config=config, attn_implementation="eager", **model_loader
    )
    if args.freeze_embeddings:
        model.base_model.embeddings.requires_grad_(False)
    if args.train_top_layers:
        if not 1 <= args.train_top_layers <= config.num_hidden_layers:
            raise ValueError("--train-top-layers exceeds the retained encoder depth")
        for layer in model.base_model.encoder.layer[: -args.train_top_layers]:
            layer.requires_grad_(False)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device)
    train_path = args.data_dir / "train.csv"
    val_path = args.data_dir / "validation.csv"
    source_text, source_y = read_csv(train_path)
    val_text, val_y = read_csv(val_path)
    train_ids = np.arange(len(source_y))
    if 0 < args.train_count < len(train_ids):
        train_ids, _ = train_test_split(
            train_ids, train_size=args.train_count, stratify=source_y, random_state=args.seed
        )
        train_ids.sort()
    train_y = source_y[train_ids]
    train_x = encode([source_text[i] for i in train_ids], tokenizer, args.max_length, args.turkish_lowercase)
    val_x = encode(val_text, tokenizer, args.max_length, args.turkish_lowercase)
    weights = (len(train_y) / (2 * np.bincount(train_y))) ** 0.5
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = math.ceil(len(train_y) / args.batch_size) * args.epochs
    warmup = max(1, int(total_steps * 0.08))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda step: min((step + 1) / warmup, max(0.1, (total_steps - step) / max(1, total_steps - warmup))),
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    history = []
    best_score = -1
    fp_dir, int_dir = [args.artifact_dir / p for p in ("bert_fp32", "bert_int8")]
    base_id = args.source_id or args.base_model
    metadata = {
        "base_model": str(base_id),
        "source_revision": args.revision,
        "turkish_lowercase": args.turkish_lowercase,
        "max_length": args.max_length,
        "encoder_layers": config.num_hidden_layers,
        "source_encoder_layers": total_layers,
        "positive_class": "OFF",
        "precision": "FP32",
    }
    # Staging prevents interrupted training from replacing a previous pair of models.
    with tempfile.TemporaryDirectory(dir=args.artifact_dir) as temp:
        staging = Path(temp)
        fp_staging, int_staging = staging / fp_dir.name, staging / int_dir.name
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            epoch_start = time.perf_counter()
            for step, (batch, targets) in enumerate(
                iter_batches(
                    train_x, train_y, rng.permutation(len(train_y)), tokenizer, args.batch_size, device
                ),
                start=1,
            ):
                optim.zero_grad(set_to_none=True)
                loss = loss_fn(model(**batch).logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                schedule.step()
                total_loss += loss.item() * len(targets)
                if step == 1 or step % 50 == 0:
                    print(
                        f"epoch {epoch}, step {step}/{math.ceil(len(train_y) / args.batch_size)}, "
                        f"elapsed {time.perf_counter() - epoch_start:.1f}s",
                        flush=True,
                    )
            probabilities = validation_probabilities(model, val_x, val_y, tokenizer, args.batch_size, device)
            cutoffs = np.linspace(0.15, 0.75, 61)
            cutoff, score = max(
                ((float(c), float(f1_score(val_y, probabilities >= c, average="macro"))) for c in cutoffs),
                key=lambda x: (x[1], -abs(x[0] - 0.5)),
            )
            history.append(
                {
                    "epoch": epoch,
                    "loss": total_loss / len(train_y),
                    "validation_macro_f1": score,
                    "threshold": cutoff,
                    "seconds": time.perf_counter() - epoch_start,
                }
            )
            print(json.dumps(history[-1]), flush=True)
            if score > best_score:
                best_score = score
                metadata.update(threshold=cutoff, best_epoch=epoch, validation_macro_f1=score)
                export(model, tokenizer, fp_staging, metadata)
        model.cpu()
        model.load_state_dict(torch.load(fp_staging / "weights.pt", map_location="cpu", weights_only=True))
        quantized = quantize_for_cpu(model)
        int_metadata = {
            **metadata,
            "precision": "INT8",
            "quantization": "dynamic signed INT8 linear weights, row-wise unsigned 8-bit embeddings; other operations float",
            "export_backend": torch.backends.quantized.engine,
        }
        export(quantized, tokenizer, int_staging, int_metadata)
        install_exports(staging, (fp_dir, int_dir))
    report = {
        **metadata,
        "train_total": len(source_y),
        "train_used": len(train_y),
        "train_subset_counts": {"NOT": int((train_y == 0).sum()), "OFF": int((train_y == 1).sum())},
        "train_subset_indices_sha256": hashlib.sha256(train_ids.astype("<i8").tobytes()).hexdigest(),
        "train_subset_index_encoding": "Sorted zero-based CSV data row indices as little-endian int64",
        "optimizer": "AdamW",
        "lr_schedule": "8% linear warmup then linear decay, minimum multiplier 0.1",
        "validation": len(val_y),
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(val_path.read_bytes()).hexdigest(),
        "seed": args.seed,
        "class_weights": weights.tolist(),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "freeze_embeddings": args.freeze_embeddings,
        "trainable_parameters": trainable_params,
        "train_top_layers": args.train_top_layers or config.num_hidden_layers,
        "history": history,
        "device": str(device),
        "torch_threads": args.threads,
        "total_seconds": time.perf_counter() - start,
    }
    (args.results_dir / "bert_training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="dbmdz/bert-base-turkish-cased")
    p.add_argument("--source-id", help="Original hub repository name when loading local files")
    p.add_argument(
        "--revision",
        default="b6e1de16c983e0f2c70664591ea3f22810072608",
        help="Exact upstream commit; use the matching SHA when changing the checkpoint",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--train-count", type=int, default=12288, help="0 uses the full train split")
    p.add_argument(
        "--turkish-lowercase",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use for Turkish uncased checkpoints, disable for cased BERTurk",
    )
    p.add_argument("--freeze-embeddings", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-top-layers", type=int, default=2, help="Freeze earlier retained encoder layers")
    p.add_argument("--layers", type=int, help="Keep only the specified first pretrained layers")
    p.add_argument("--max-length", type=int, default=48)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.train_count < 0 or min(args.batch_size, args.epochs, args.threads, args.max_length) <= 0:
        p.error(
            "train-count must be nonnegative; batch-size, epochs, threads, and max-length must be positive"
        )
    main(args)
