"""CPU inference using the exported Turkish BERT INT8 ONNX graph."""

import json
from pathlib import Path

import onnxruntime as ort
from scipy.special import softmax
from transformers import AutoTokenizer

from bert_model import prepare_text


class BertOnnxModerator:
    score_kind = "transformer_class_probability"

    def __init__(self, folder, threads=1):
        folder = Path(folder)
        self.metadata = json.loads((folder / "moderation.json").read_text(encoding="utf-8"))
        self.threshold = self.metadata["threshold"]
        self.max_length = self.metadata["max_length"]
        self.turkish_lowercase = self.metadata["turkish_lowercase"]
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(folder / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_names = [tensor.name for tensor in self.session.get_inputs()]

    def probabilities(self, texts):
        tokens = self.tokenizer(
            [prepare_text(text, self.turkish_lowercase) for text in texts],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        logits, = self.session.run(["logits"], {name: tokens[name] for name in self.input_names})
        return softmax(logits, axis=1)[:, 1].tolist()

    def predict_one(self, text):
        probability = self.probabilities([text])[0]
        flag = int(probability >= self.threshold)
        return flag, probability if flag else 1 - probability

    def predict_many(self, texts, batch_size=32):
        return [
            int(probability >= self.threshold)
            for i in range(0, len(texts), batch_size)
            for probability in self.probabilities(texts[i : i + batch_size])
        ]
