"""Interactive CPU inference for Turkish chat moderation."""

from functools import lru_cache
from html import escape
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter_ns
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from classical import load_model
from neural import NeuralModerator


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
# The first installed model is the default in the chat.
MODEL_MENU = {
    "bert_int8": ("BERTurk · INT8", "BERTurk INT8", "bert_int8/moderation.json"),
    "bert_fp32": ("BERTurk · FP32", "BERTurk FP32", "bert_fp32/moderation.json"),
    "tfidf": ("TF-IDF + lojistik regresyon", "TF-IDF", "tfidf.joblib"),
    "rules": ("Sözlük + düzenleme mesafesi", "Sözlük", "rules.json"),
    "neural": ("Word2Vec + sinir ağı", "Word2Vec", "neural.npz"),
}
app = FastAPI(title="Türkçe sohbet moderasyonu")


class ChatInput(BaseModel):
    model: Literal["tfidf", "rules", "neural", "bert_fp32", "bert_int8"]
    text: str = Field(min_length=1, max_length=1500)


def installed(name):
    marker = ARTIFACTS / MODEL_MENU[name][2]
    if name.startswith("bert_"):
        folder = marker.parent
        onnx = (folder / "model.onnx").is_file()
        weights_ready = (find_spec("onnxruntime") is not None if onnx
                         else (folder / "weights.pt").is_file())
        return (marker.is_file() and weights_ready
                and (folder / "config.json").is_file()
                and any((folder / f).is_file() for f in ("tokenizer.json", "vocab.txt"))
                and find_spec("torch") is not None and find_spec("transformers") is not None)
    return marker.is_file()


@lru_cache(maxsize=len(MODEL_MENU))
def get_predictor(name):
    if name.startswith("bert_"):
        if not installed(name):
            raise HTTPException(503, "Transformer bağımlılıkları ve model dosyaları gerekli.")
        import torch
        from bert_model import load_deployed_bert, CPU_THREADS

        torch.set_num_threads(CPU_THREADS)
        return load_deployed_bert(ARTIFACTS / name)
    if name == "neural":
        return NeuralModerator(ARTIFACTS / "neural.npz")
    return load_model(name, ARTIFACTS)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    template = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    options = [
        f'<option value="{name}" data-short="{escape(short)}">{escape(label)}</option>'
        for name, (label, short, _) in MODEL_MENU.items() if installed(name)
    ]
    return template.replace("<!-- MODEL_OPTIONS -->", "\n".join(options))


@app.post("/predict")
def predict(message: ChatInput):
    text = message.text.strip()
    if not text:
        raise HTTPException(422, "Bir mesaj yazın.")
    try:
        predictor = get_predictor(message.model)
    except FileNotFoundError:
        script = "train_neural.py" if message.model == "neural" else "train_classical.py"
        raise HTTPException(503, f"Model dosyası bulunamadı. Önce python {script} komutunu çalıştırın.") from None
    start = perf_counter_ns()
    label, confidence = predictor.predict_one(text)
    latency_ms = (perf_counter_ns() - start) / 1_000_000
    flagged = bool(label)
    result = {
        "model": message.model,
        "text": "Mesaj maskelendi" if flagged else text,
        "flagged": flagged,
        "confidence": confidence,
        "score_kind": predictor.score_kind,
        "latency_ms": latency_ms,
    }
    if hasattr(predictor, "threshold"):
        result["offensive_probability"] = confidence if flagged else 1 - confidence
        result["decision_threshold"] = predictor.threshold
    return result
