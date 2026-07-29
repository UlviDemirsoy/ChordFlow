"""En son el PyTorch checkpointini tarayıcı uyumlu ONNX'e dönüştürür."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch
from torch import nn

from left_hand_model_training import INPUT_SIZE, LeftHandMLP


class BrowserInferenceModel(nn.Module):
    """Ham 21x3 landmark girdisini normalize eder ve 6 logit üretir."""

    def __init__(
        self,
        classifier: LeftHandMLP,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.register_buffer("feature_mean", feature_mean.float().reshape(1, -1))
        self.register_buffer("feature_std", feature_std.float().reshape(1, -1))

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        landmarks = raw_features.reshape(-1, 21, 3)
        landmarks = landmarks - landmarks[:, 0:1, :]

        palm_vector = landmarks[:, 9, :]
        palm_scale = torch.linalg.vector_norm(palm_vector, dim=1)
        fallback_scale = torch.linalg.vector_norm(landmarks, dim=2).amax(dim=1)
        palm_scale = torch.where(
            palm_scale > 1e-6,
            palm_scale,
            fallback_scale,
        ).clamp_min(1e-6)
        landmarks = landmarks / palm_scale[:, None, None]

        palm_x = landmarks[:, 9, 0]
        palm_y = landmarks[:, 9, 1]
        xy_norm = torch.sqrt(palm_x.square() + palm_y.square()).clamp_min(1e-6)
        cosine = -palm_y / xy_norm
        sine = -palm_x / xy_norm

        x_values = landmarks[:, :, 0]
        y_values = landmarks[:, :, 1]
        rotated_x = cosine[:, None] * x_values - sine[:, None] * y_values
        rotated_y = sine[:, None] * x_values + cosine[:, None] * y_values
        landmarks = torch.stack(
            (rotated_x, rotated_y, landmarks[:, :, 2]),
            dim=2,
        ).clamp(-4.0, 4.0)

        features = landmarks.reshape(-1, INPUT_SIZE)
        standardized = (features - self.feature_mean) / self.feature_std
        return self.classifier(standardized)


def latest_checkpoint(training_root: Path, hand: str = "left") -> Path:
    checkpoints = list(training_root.glob(f"run_*/{hand}_hand_model.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"Checkpoint bulunamadı: {training_root}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="El modelini ONNX'e çevirir.")
    parser.add_argument("--hand", choices=("left", "right"), default="left")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Dönüştürülecek left_hand_model.pt. Verilmezse en güncel run kullanılır.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint
        else latest_checkpoint(
            project_root / "model" / "training_outputs" / f"{args.hand}_hand",
            args.hand,
        )
    )
    output_path = (
        args.output.resolve()
        if args.output
        else project_root
        / "public"
        / "models"
        / f"{args.hand}_hand_model.onnx"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classifier = LeftHandMLP(
        dropout=float(checkpoint["dropout"]),
        class_count=len(checkpoint["class_labels"]),
    )
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()

    browser_model = BrowserInferenceModel(
        classifier,
        checkpoint["feature_mean"],
        checkpoint["feature_std"],
    ).eval()

    dummy_input = torch.zeros(1, INPUT_SIZE, dtype=torch.float32)
    torch.onnx.export(
        browser_model,
        dummy_input,
        output_path,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={
            "landmarks": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=18,
        dynamo=False,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    metadata = {
        "source_checkpoint": str(checkpoint_path.relative_to(project_root)),
        "input_name": "landmarks",
        "input_shape": ["batch", 63],
        "output_name": "logits",
        "class_labels": checkpoint["class_labels"],
        "hand": args.hand,
        "preprocessing_embedded": True,
        "recommended_confidence_threshold": 0.85,
        "recommended_margin_threshold": 0.20,
        "note": (
            "Class 0 explicit unknown sınıfıdır. Confidence/margin eşikleri "
            "ayrıca kararsız/rejected sonuç üretir."
            if 0 in checkpoint["class_labels"]
            else (
                "Modelde unknown sınıfı yoktur. Eşik altındaki sonuçlar yalnızca "
                "kararsız/rejected olarak yorumlanır."
            )
        ),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"ONNX: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Source: {checkpoint_path}")


if __name__ == "__main__":
    main()
