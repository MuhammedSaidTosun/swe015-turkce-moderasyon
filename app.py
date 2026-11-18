"""Interactive CPU inference for Turkish chat moderation."""

from functools import lru_cache
from pathlib import Path
from time import perf_counter_ns
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from classical import load_model
from neural import NeuralModerator


ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Türkçe sohbet moderasyonu")


class ChatInput(BaseModel):
    model: Literal["tfidf", "rules", "neural"]
    text: str = Field(min_length=1, max_length=1500)


@lru_cache(maxsize=3)
def get_predictor(name):
    if name == "neural":
        return NeuralModerator(ROOT / "artifacts" / "neural.npz")
    return load_model(name, ROOT / "artifacts")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html", media_type="text/html")


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
    if message.model == "neural":
        result["offensive_probability"] = confidence if flagged else 1 - confidence
        result["decision_threshold"] = predictor.threshold
    return result
