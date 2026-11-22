"""Inference for the Word2Vec and feedforward Turkish message classifier."""

import numpy as np

from classical import normalize_text


def pool_tokens(tokens, token_index, vectors):
    """Mean of known token vectors; an all-unknown message has a zero vector."""
    indices = [token_index[token] for token in tokens if token in token_index]
    if not indices:
        return np.zeros(vectors.shape[1], dtype=vectors.dtype)
    return vectors[indices].mean(axis=0)


def lexical_terms(tokens):
    """Word unigrams, ordered pairs and within-word pieces, including suffixes."""
    for token in tokens:
        yield "w:" + token
        bounded = "^" + token + "$"
        for width in (3, 4):
            for start in range(len(bounded) - width + 1):
                yield bounded[start:start + width]
    for left, right in zip(tokens, tokens[1:]):
        yield left + " " + right


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
            self.term_index = None
            if "terms" in data:
                self.term_index = {term: i for i, term in enumerate(data["terms"].tolist())}
                self.term_weights = data["term_weights"]
                self.term_gain = data["term_gain"].item()
                self.dense_gain = data["dense_gain"].item()

    def _offensive_probability(self, text):
        tokens = normalize_text(text).split()
        features = pool_tokens(tokens, self.token_index, self.vectors)
        standardized = (features - self.mean) / self.scale
        if self.term_index is not None:
            standardized *= self.dense_gain
        activation = standardized @ self.w1 + self.b1
        if self.term_index is not None:
            indices = sorted({self.term_index[t] for t in lexical_terms(tokens) if t in self.term_index})
            if indices:
                activation += self.term_weights[indices].sum(axis=0) * (self.term_gain / np.sqrt(len(indices)))
        hidden = np.maximum(activation, 0)
        logits = hidden @ self.w2 + self.b2
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))

    def predict_one(self, text):
        probability = float(self._offensive_probability(text))
        label = int(probability >= self.threshold)
        return label, probability if label else 1 - probability

    def predict_many(self, texts):
        return np.fromiter((self.predict_one(text)[0] for text in texts), dtype=np.int64)
