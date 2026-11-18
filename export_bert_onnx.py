"""Export a fine-tuned FP32 deployment directory as a standalone BERT INT8 ONNX model."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import onnx
import onnxruntime
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn

from bert_model import BertModerator


class BertLogits(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        # An additive mask keeps padded tokens masked for symbolic batch and sequence lengths.
        mask = (1 - attention_mask[:, None, None, :].to(torch.float32)) * torch.finfo(torch.float32).min
        return self.model(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids).logits


def export_fp32_graph(moderator, fp_onnx):
    """Export BERT logits with dynamic batch and sequence axes."""
    wrapped = BertLogits(moderator.model).eval()
    example = moderator.tokenizer(["Merhaba nasılsın?", "Güzel oyun."], return_tensors="pt", padding=True)
    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    inputs = tuple(example[name] for name in input_names)
    axes = {name: {0: "batch_size", 1: "sequence_length"} for name in input_names}
    axes["logits"] = {0: "batch_size"}
    with torch.inference_mode():
        torch.testing.assert_close(
            wrapped(*inputs), moderator.model(**example).logits, rtol=1e-5, atol=2e-5
        )
        torch.onnx.export(
            wrapped, inputs, str(fp_onnx), input_names=input_names, output_names=["logits"],
            dynamic_axes=axes, opset_version=17, dynamo=False, external_data=False,
        )


def export_onnx_int8(fp_folder, destination):
    """Write a new self-contained export; a failed conversion leaves no partial destination."""
    fp_folder, destination = Path(fp_folder), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Choose an unused destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    moderator = BertModerator(fp_folder)
    if moderator.metadata["precision"] != "FP32":
        raise ValueError("INT8 conversion requires the selected FP32 checkpoint.")
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        root = Path(tmp)
        deployed = root / "int8"
        deployed.mkdir()
        fp_onnx = root / "fp32.onnx"
        export_fp32_graph(moderator, fp_onnx)
        quantize_dynamic(
            str(fp_onnx), str(deployed / "model.onnx"), weight_type=QuantType.QInt8,
            per_channel=True, op_types_to_quantize=["MatMul", "Gemm", "Gather"],
            extra_options={"MatMulConstBOnly": True},
        )
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            shutil.copyfile(fp_folder / name, deployed / name)
        graph = onnx.load(str(deployed / "model.onnx")).graph
        dense_ops = sum(node.op_type == "MatMulInteger" for node in graph.node)
        if not dense_ops:
            raise ValueError("Exported graph has no integer matrix multiplications.")
        meta = dict(
            moderator.metadata, precision="INT8", export_backend="ONNX Runtime CPU",
            onnxruntime_version=onnxruntime.__version__, opset=17, int8_dense_ops=dense_ops,
            quantization="Dynamic INT8 per-channel dense weights, UINT8 per-tensor embedding tables; remaining operations floating point.",
        )
        (deployed / "moderation.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        deployed.replace(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fp_folder", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    export_onnx_int8(args.fp_folder, args.destination)
