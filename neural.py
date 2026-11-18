"""Inference for the Word2Vec and feedforward Turkish message classifier."""

import numpy as np

from classical import normalize_text


def pool_tokens(tokens, token_index, vectors):
    """Mean of known token vectors; an all-unknown message has a zero vector."""
    indices = [token_index[token] for token in tokens if token in token_index]
    if not indices:
        return np.zeros(vectors.shape[1], dtype=vectors.dtype)
    return vectors[indices].mean(axis=0)


class NeuralModerator:
    score_kind = "neural_class_probability"

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.token_index = {token: i for i, token in enumerate(data["vocab"].tolist())}
            self.vectors = data["vectors"]
            self.mean = data["mean"]
            self.scale = data["scale"]
            self.w1 = data["w1"]
            self.b1 = data["b1"]
            self.w2 = data["w2"].reshape(-1)
            self.b2 = data["b2"].item()
            self.threshold = float(data["threshold"].item())

    def _encode(self, text):
        return pool_tokens(normalize_text(text).split(), self.token_index, self.vectors)

    def _offensive_probability(self, features):
        standardized = (features - self.mean) / self.scale
        hidden = np.maximum(standardized @ self.w1 + self.b1, 0)
        logits = hidden @ self.w2 + self.b2
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))

    def predict_one(self, text):
        probability = float(self._offensive_probability(self._encode(text)))
        label = int(probability >= self.threshold)
        return label, probability if label else 1 - probability

    def predict_many(self, texts):
        vectors = [self._encode(text) for text in texts]
        if not vectors:
            return np.empty(0, dtype=np.int64)
        probabilities = self._offensive_probability(np.stack(vectors))
        return (probabilities >= self.threshold).astype(np.int64)
