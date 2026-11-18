"""Train Turkish CBOW Word2Vec on train messages, then fit a feedforward classifier."""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from classical import normalize_text
from neural import pool_tokens


SEED = 42
DIMENSIONS = 48
CONTEXT_RADIUS = 3
NEGATIVES = 5
MIN_FREQUENCY = 2
EMBEDDING_EPOCHS = 12
EMBEDDING_BATCH = 256
SUBSAMPLE = 0.003
HIDDEN_UNITS = 64
MAX_CLASSIFIER_EPOCHS = 50
PATIENCE = 8
THRESHOLDS = np.linspace(0.15, 0.75, 61)


def read_split(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    tokens = [normalize_text(row["text"]).split() for row in rows]
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError(f"{path.name} must contain both labels 0 and 1")
    return tokens, labels


def make_contexts(sentences, keep, padding, rng):
    """Create context words and central targets without crossing message boundaries."""
    n_rows = sum(map(len, sentences))
    contexts = np.full((n_rows, 2 * CONTEXT_RADIUS), padding, dtype=np.int32)
    targets = np.empty(n_rows, dtype=np.int32)
    offsets = [d for d in range(-CONTEXT_RADIUS, CONTEXT_RADIUS + 1) if d != 0]
    start = 0
    for sentence in sentences:
        ids = sentence[rng.random(len(sentence)) < keep[sentence]]
        n = len(ids)
        if n < 2:
            continue
        targets[start:start + n] = ids
        for col, d in enumerate(offsets):
            if 0 < d < n:
                contexts[start:start + n - d, col] = ids[d:]
            elif 0 < -d < n:
                contexts[start - d:start + n, col] = ids[:n + d]
        start += n
    if start == 0:
        raise ValueError("No pairs of words were available to train Word2Vec.")
    return contexts[:start], targets[:start]


def fit_word2vec(corpus, epochs, seed):
    """CBOW estimates the central word from the average of its neighbours."""
    frequencies = Counter(word for sentence in corpus for word in sentence)
    vocab = sorted(w for w, count in frequencies.items() if count >= MIN_FREQUENCY)
    if len(vocab) < 2:
        raise ValueError("Word2Vec requires at least two words meeting min_frequency")
    ids = {word: i for i, word in enumerate(vocab)}
    sentences = [np.array([ids[w] for w in sentence if w in ids], dtype=np.int32)
                 for sentence in corpus]
    counts = np.array([frequencies[word] for word in vocab], dtype=np.float64)
    frequency = counts / counts.sum()
    keep = np.minimum(1, (np.sqrt(frequency / SUBSAMPLE) + 1) * SUBSAMPLE / frequency)
    noise_cdf = np.cumsum(counts ** 0.75)
    noise_cdf /= noise_cdf[-1]
    rng = np.random.default_rng(seed)
    padding = len(vocab)
    inputs = rng.uniform(-0.5 / DIMENSIONS, 0.5 / DIMENSIONS,
                         (len(vocab) + 1, DIMENSIONS)).astype(np.float32)
    inputs[padding] = 0
    outputs = np.zeros((len(vocab), DIMENSIONS), dtype=np.float32)
    losses = []
    for epoch in range(epochs):
        contexts, targets = make_contexts(sentences, keep, padding, rng)
        shuffled = rng.permutation(len(targets))
        total_loss = 0.
        for begin in range(0, len(targets), EMBEDDING_BATCH):
            sel = shuffled[begin:begin + EMBEDDING_BATCH]
            words = contexts[sel]
            valid = words != padding
            n_context = valid.sum(axis=1).astype(np.float32)[:, None]
            centers = targets[sel]
            noise = np.searchsorted(noise_cdf, rng.random((len(sel), NEGATIVES)))
            candidates = np.column_stack([centers, noise])
            hidden = inputs[words].sum(axis=1) / n_context
            out_vectors = outputs[candidates]
            logits = np.einsum("bd,bkd->bk", hidden, out_vectors)
            active = noise != centers[:, None]
            total_loss += (np.logaddexp(0, -logits[:, 0]).sum()
                           + (np.logaddexp(0, logits[:, 1:]) * active).sum())
            # Gradient ascent on log(sigmoid(positive)) + sum(log(sigmoid(-negative))).
            progress = (epoch + begin / len(targets)) / epochs
            lr = np.float32(0.07 * (1 - progress) + 0.008 * progress)
            delta = -expit(logits)
            delta[:, 0] += 1
            delta[:, 1:] *= active
            delta *= lr
            hidden_update = np.einsum("bk,bkd->bd", delta, out_vectors) / n_context
            np.add.at(outputs, candidates.ravel(),
                      (delta[:, :, None] * hidden[:, None, :]).reshape(-1, DIMENSIONS))
            for col in range(words.shape[1]):
                mask = valid[:, col]
                np.add.at(inputs, words[mask, col], hidden_update[mask])
        losses.append(float(total_loss / len(targets)))
        print(f"CBOW {epoch + 1}/{epochs}: loss per target {losses[-1]:.4f}", flush=True)
    stats = {"method": "CBOW negative sampling", "dimensions": DIMENSIONS,
             "context_radius": CONTEXT_RADIUS, "negatives": NEGATIVES,
             "negative_distribution": "train token count raised to 0.75",
             "min_frequency": MIN_FREQUENCY, "subsampling": SUBSAMPLE,
             "batch_size": EMBEDDING_BATCH, "epochs": epochs, "vocab_size": len(vocab),
             "training_tokens": sum(frequencies.values()),
             "training_tokens_in_vocab": int(counts.sum()), "epoch_loss": losses}
    return vocab, ids, inputs[:-1], stats


def select_threshold(labels, probabilities):
    predicted = probabilities[:, None] >= THRESHOLDS[None, :]
    positive = labels[:, None] == 1
    tp = (predicted & positive).sum(axis=0)
    fp = (predicted & ~positive).sum(axis=0)
    fn = positive.sum() - tp
    tn = len(labels) - tp - fp - fn
    scores = (tp / (2 * tp + fp + fn) + tn / (2 * tn + fp + fn))
    # Equal scores prefer the cutoff nearest 0.5.
    tied = np.flatnonzero(scores == scores.max())
    best = tied[np.argmin(np.abs(THRESHOLDS[tied] - 0.5))]
    return float(THRESHOLDS[best]), float(scores[best])


def fit_classifier(train_x, train_y, val_x, val_y, seed):
    clf = MLPClassifier(hidden_layer_sizes=(HIDDEN_UNITS,), activation="relu", solver="sgd",
                        batch_size=128, learning_rate_init=0.018, momentum=0.9, alpha=0.001,
                        random_state=np.random.RandomState(seed), shuffle=True)
    # Class-frequency weighting is estimated exclusively on the training split.
    class_weight = np.sqrt(len(train_y) / (2 * np.bincount(train_y, minlength=2)))
    sample_weights = class_weight[train_y]
    best_score, best_epoch = -1., 0
    history = []
    for epoch in range(1, MAX_CLASSIFIER_EPOCHS + 1):
        clf.partial_fit(train_x, train_y, classes=[0, 1], sample_weight=sample_weights)
        scores = clf.predict_proba(val_x)[:, 1]
        threshold, score = select_threshold(val_y, scores)
        history.append({"epoch": epoch, "loss": float(clf.loss_), "val_macro_f1": score,
                        "threshold": threshold})
        if score > best_score:
            best_score, best_epoch = score, epoch
            params = dict(w1=clf.coefs_[0].copy(), b1=clf.intercepts_[0].copy(),
                          w2=clf.coefs_[1][:, 0].copy(), b2=clf.intercepts_[1][0],
                          threshold=np.float64(threshold))
        if epoch - best_epoch >= PATIENCE:
            break
    stats = {"representation": "mean of in-vocabulary word vectors, standardized on train",
             "method": "one ReLU hidden layer, sigmoid output",
             "optimizer": "minibatch SGD with Nesterov momentum", "learning_rate": 0.018,
             "momentum": 0.9, "alpha": 0.001, "batch_size": 128, "hidden_units": HIDDEN_UNITS,
             "train_class_weights": class_weight.tolist(), "best_epoch": best_epoch,
             "epochs_run": len(history), "threshold": float(params["threshold"]),
             "validation_macro_f1": best_score, "validation_history": history}
    return params, stats


def train(args):
    train_file = args.data_dir / "train.csv"
    val_file = args.data_dir / "validation.csv"
    train_tokens, train_y = read_split(train_file)
    val_tokens, val_y = read_split(val_file)
    start = perf_counter()
    vocab, word_ids, vectors, word_stats = fit_word2vec(train_tokens, args.embedding_epochs, args.seed)
    train_x = np.stack([pool_tokens(row, word_ids, vectors) for row in train_tokens])
    val_x = np.stack([pool_tokens(row, word_ids, vectors) for row in val_tokens])
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.
    params, clf_stats = fit_classifier((train_x - mean) / scale, train_y,
                                       (val_x - mean) / scale, val_y, args.seed)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    report = {"seed": args.seed, "rows": {"train": len(train_y), "validation": len(val_y)},
              "split_sha256": {p.stem: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (train_file, val_file)},
              "word2vec": word_stats, "classifier": clf_stats,
              "training_seconds": perf_counter() - start}
    temp = args.artifact_dir / f"neural-{os.getpid()}.npz"
    try:
        np.savez_compressed(temp, vocab=np.array(vocab), vectors=vectors, mean=mean, scale=scale,
                            **params)
        temp.replace(args.artifact_dir / "neural.npz")
    finally:
        temp.unlink(missing_ok=True)
    (args.results_dir / "neural_training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Selected epoch {clf_stats['best_epoch']}, threshold {clf_stats['threshold']:.2f}, "
          f"validation macro-F1 {clf_stats['validation_macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--embedding-epochs", type=int, default=EMBEDDING_EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.embedding_epochs < 1:
        parser.error("--embedding-epochs must be positive")
    with threadpool_limits(limits=1):
        train(args)
