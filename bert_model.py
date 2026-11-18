"""CPU deployment of the fine-tuned Turkish BERT classifier and its INT8 copy."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.ao.quantization import (
    default_dynamic_qconfig,
    float_qparams_weight_only_qconfig,
    quantize_dynamic,
)
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

CPU_THREADS = 1


def load_deployed_bert(folder, threads=CPU_THREADS):
    """Load the independently deployable PyTorch or ONNX model in this directory."""
    folder = Path(folder)
    if (folder / "model.onnx").is_file():
        from bert_onnx import BertOnnxModerator

        return BertOnnxModerator(folder, threads=threads)
    return BertModerator(folder)


def prepare_text(text, turkish_lowercase):
    return text.translate(str.maketrans({"I": "ı", "İ": "i"})).lower() if turkish_lowercase else text


def quantize_for_cpu(model):
    """Dynamic INT8 linear layers and row-wise 8-bit weights for embedding tables."""
    supported = torch.backends.quantized.supported_engines
    engine = next((name for name in ("x86", "fbgemm", "qnnpack") if name in supported), None)
    if engine is None:
        raise RuntimeError("This PyTorch build lacks an x86, fbgemm or qnnpack CPU quantization backend.")
    torch.backends.quantized.engine = engine
    return quantize_dynamic(
        model.cpu().eval(),
        {
            nn.Linear: default_dynamic_qconfig,
            nn.Embedding: float_qparams_weight_only_qconfig,
        },
        inplace=True,
    )


class BertModerator:
    score_kind = "transformer_class_probability"

    def __init__(self, folder):
        folder = Path(folder)
        self.metadata = json.loads((folder / "moderation.json").read_text(encoding="utf-8"))
        self.threshold = float(self.metadata["threshold"])
        self.max_length = self.metadata["max_length"]
        self.turkish_lowercase = self.metadata["turkish_lowercase"]
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        config = AutoConfig.from_pretrained(folder, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_config(
            config, attn_implementation="eager"
        ).eval()
        if self.metadata["precision"] == "INT8":
            quantize_for_cpu(self.model)
        state = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def probabilities(self, texts):
        tokens = self.tokenizer(
            [prepare_text(t, self.turkish_lowercase) for t in texts],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        return self.model(**tokens).logits.softmax(-1)[:, 1].tolist()

    def predict_one(self, text):
        prob = self.probabilities([text])[0]
        flag = int(prob >= self.threshold)
        return flag, prob if flag else 1 - prob

    def predict_many(self, texts, batch_size=32):
        return [
            int(p >= self.threshold)
            for i in range(0, len(texts), batch_size)
            for p in self.probabilities(texts[i : i + batch_size])
        ]
