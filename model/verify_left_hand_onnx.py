"""ONNX çıktısını aynı checkpointin PyTorch çıktısıyla karşılaştırır."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from onnx.reference import ReferenceEvaluator

from export_left_hand_onnx import BrowserInferenceModel, latest_checkpoint
from left_hand_model_training import (
    INPUT_SIZE,
    LeftHandMLP,
    landmark_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="El ONNX modelini doğrular.")
    parser.add_argument("--hand", choices=("left", "right"), default="left")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = latest_checkpoint(
        project_root / "model" / "training_outputs" / f"{args.hand}_hand",
        args.hand,
    )
    onnx_path = (
        project_root
        / "public"
        / "models"
        / f"{args.hand}_hand_model.onnx"
    )
    csv_path = (
        project_root
        / "model"
        / "data"
        / args.hand
        / "landmarks.csv"
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    class_labels = checkpoint["class_labels"]
    classifier = LeftHandMLP(
        dropout=float(checkpoint["dropout"]),
        class_count=len(class_labels),
    )
    classifier.load_state_dict(checkpoint["model_state_dict"])
    pytorch_model = BrowserInferenceModel(
        classifier.eval(),
        checkpoint["feature_mean"],
        checkpoint["feature_std"],
    ).eval()

    frame = pd.read_csv(csv_path, nrows=32)
    raw_features = frame[landmark_columns()].to_numpy(dtype=np.float32)

    with torch.no_grad():
        pytorch_logits = pytorch_model(torch.from_numpy(raw_features)).numpy()

    evaluator = ReferenceEvaluator(str(onnx_path))
    onnx_logits = np.asarray(
        evaluator.run(None, {"landmarks": raw_features})[0],
        dtype=np.float32,
    ).reshape(-1, len(class_labels))

    maximum_difference = float(np.max(np.abs(pytorch_logits - onnx_logits)))
    prediction_agreement = float(
        np.mean(pytorch_logits.argmax(axis=1) == onnx_logits.argmax(axis=1))
    )

    print(f"Input shape: {raw_features.shape}")
    print(f"Output shape: {onnx_logits.shape}")
    print(f"Max absolute logit difference: {maximum_difference:.8f}")
    print(f"Prediction agreement: {prediction_agreement:.2%}")

    if prediction_agreement != 1.0 or maximum_difference > 1e-4:
        raise RuntimeError("ONNX ve PyTorch çıktıları eşleşmiyor.")


if __name__ == "__main__":
    main()
