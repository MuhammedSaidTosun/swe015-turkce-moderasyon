"""Classical predictors for Turkish chat moderation (0 = clean, 1 = offensive)."""

import json
import re
import unicodedata
from pathlib import Path


DEFAULT_LEXICON = (
    "ahmak", "ahmağın", "ahmaksın", "aptal", "aptallar", "aptalsın",
    "dangalağın", "dangalak", "dangalaksın", "embesil", "geri zekalı", "gerizekalı",
    "gerizekalısın", "haysiyetsiz", "haysiyetsizsin", "hıyar", "hıyarsın",
    "it oğlu it", "kaltak", "mal mısın", "malsın", "manyak", "manyaklar",
    "namussuz", "namussuzsun", "orospu", "piç", "piçler", "puşt", "salak",
    "salaklar", "salaksın", "şerefsizin", "siktir", "siktirin", "sikik",
    "sürtük", "şerefsiz", "şerefsizler", "şerefsizsin", "yavşak",
    "yavşaklar", "yavşaksın", "zekasız", "zekasızsın",
)
TURKISH_CASE = str.maketrans({"I": "ı", "İ": "i"})
ASCII_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})
URL = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION = re.compile(r"(?<!\w)@\w+")
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
REPEATED_LETTER = re.compile(r"([^\W\d_])\1{2,}")


def normalize_text(text):
    """Keep Turkish letters; regularise case, links, mentions and noisy spellings."""
    text = MENTION.sub(" ", URL.sub(" ", unicodedata.normalize("NFKC", text)))
    text = text.translate(TURKISH_CASE).lower()
    words = []
    for token in TOKEN.findall(text):
        if any(char.isalpha() for char in token) and any(char.isdigit() for char in token):
            token = token.translate(LEET)
        words.append(REPEATED_LETTER.sub(r"\1", token))
    return " ".join(words)


def _deletions(word):
    return {word[:i] + word[i + 1:] for i in range(len(word))}


def _one_edit(left, right):
    """Whether the Levenshtein distance of two different strings is exactly one."""
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) == 1
    if len(left) > len(right):
        left, right = right, left
    if len(right) != len(left) + 1:
        return False
    i = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), len(left))
    return left[i:] == right[i + 1:]


class RuleModerator:
    score_kind = "heuristic"

    def __init__(self, lexicon=DEFAULT_LEXICON):
        self.lexicon = tuple(lexicon)
        folded = {normalize_text(word).translate(ASCII_FOLD) for word in self.lexicon}
        self.words = {word for word in folded if " " not in word}
        self.phrases = {" " + word + " " for word in folded if " " in word}
        self.index = {}
        # Short fuzzy matches (e.g. 'salak' vs 'salat') cause frequent false alarms.
        # Restrict one-edit matching to longer entries. Exact matches have no length limit.
        for word in self.words:
            if len(word) >= 6:
                for key in _deletions(word) | {word}:
                    self.index.setdefault(key, set()).add(word)
        self.max_length = max(map(len, self.words), default=0)

    def predict_one(self, text):
        normalized = normalize_text(text).translate(ASCII_FOLD)
        tokens = normalized.split()
        padded = " " + normalized + " "
        if any(token in self.words for token in tokens) or any(phrase in padded for phrase in self.phrases):
            return 1, 0.95
        for token in tokens:
            if 5 <= len(token) <= self.max_length + 1:
                candidates = set()
                for key in _deletions(token) | {token}:
                    candidates.update(self.index.get(key, ()))
                if any(_one_edit(token, candidate) for candidate in candidates):
                    return 1, 0.75
        # These three hand-set scores indicate rule evidence, not probabilities.
        return 0, 0.55

    def save(self, path):
        Path(path).write_text(json.dumps({"lexicon": self.lexicon}, ensure_ascii=False), encoding="utf-8")


class TfidfModerator:
    score_kind = "logistic_regression_probability"

    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict_one(self, text):
        probabilities = self.classifier.predict_proba(self.vectorizer.transform([text]))[0]
        index = int(probabilities.argmax())
        return int(self.classifier.classes_[index]), float(probabilities[index])

    def predict_many(self, texts):
        return self.classifier.predict(self.vectorizer.transform(texts))


def load_model(name, artifact_dir="artifacts"):
    directory = Path(artifact_dir)
    if name == "rules":
        settings = json.loads((directory / "rules.json").read_text(encoding="utf-8"))
        return RuleModerator(settings["lexicon"])
    if name == "tfidf":
        import joblib

        return joblib.load(directory / "tfidf.joblib")
    raise ValueError(f"Unknown classical model: {name}")
