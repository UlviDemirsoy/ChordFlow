"""
ChordFlow sol el unknown + derece sınıflandırıcısı eğitim scripti.

Mimari
-------
Input (63) -> Linear(128) -> BatchNorm -> ReLU -> Dropout
           -> Linear(64)  -> ReLU      -> Dropout
           -> Linear(8 logits)

Normalizasyon
-------------
1. Her landmark için bilek (landmark 0) koordinatı çıkarılır.
2. Tüm koordinatlar bilek ile middle MCP (landmark 9) mesafesine bölünür.
3. El, bilek -> middle MCP doğrultusu ekranda yukarı bakacak şekilde XY
   düzleminde döndürülür.
4. Sadece train splitinden hesaplanan feature mean/std ile standardize edilir.

Split
-----
Kamera burstlerindeki ardışık kareler neredeyse aynıdır. Bu nedenle satırları
doğrudan rastgele bölmek yerine her sınıf zaman sırasına alınır ve temporal
bloklara ayrılır. Bloklar train/validation/test arasında paylaştırılır.

Çıktılar
--------
Her çalıştırma model/training_outputs/left_hand/run_<timestamp> altında model,
preprocessing değerleri, metrikler ve grafikler oluşturur.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


LANDMARK_COUNT = 21
COORDINATE_COUNT = 3
INPUT_SIZE = LANDMARK_COUNT * COORDINATE_COUNT
CLASS_LABELS = [0, 1, 2, 3, 4, 5, 6, 7]
CLASS_NAMES = ["unknown", *[f"degree_{label}" for label in CLASS_LABELS[1:]]]


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.25
    patience: int = 25
    temporal_block_size: int = 20
    temporal_gap_seconds: float = 2.0
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    augmentation_noise_std: float = 0.015


class LandmarkDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        augment: bool,
        noise_std: float,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features[index].clone()
        if self.augment and self.noise_std > 0:
            features += torch.randn_like(features) * self.noise_std
        return features, self.labels[index]


class LeftHandMLP(nn.Module):
    def __init__(
        self,
        dropout: float = 0.25,
        class_count: int = len(CLASS_LABELS),
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(64, class_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="ChordFlow sol el landmark modelini eğitir."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=project_root / "model" / "data" / "left" / "landmarks.csv",
        help="Dataset CSV yolu.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "model" / "training_outputs" / "left_hand",
        help="Eğitim çıktılarının ana klasörü.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def landmark_columns() -> list[str]:
    return [
        f"{axis}{index}"
        for index in range(LANDMARK_COUNT)
        for axis in ("x", "y", "z")
    ]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset bulunamadı: {csv_path}")

    frame = pd.read_csv(csv_path)
    required_columns = {
        "id",
        "label",
        "captured_at",
        "handedness_score",
        *landmark_columns(),
    }
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV kolonları eksik: {sorted(missing)}")

    frame = frame.copy()
    frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True)
    frame["label"] = frame["label"].astype(int)
    frame = frame[frame["label"].isin(CLASS_LABELS)].reset_index(drop=True)

    if frame.empty:
        raise ValueError("CSV içinde eğitilecek örnek bulunamadı.")

    invalid_features = ~np.isfinite(
        frame[landmark_columns()].to_numpy(dtype=np.float32)
    ).all(axis=1)
    if invalid_features.any():
        print(f"Uyarı: {invalid_features.sum()} geçersiz satır çıkarıldı.")
        frame = frame.loc[~invalid_features].reset_index(drop=True)

    class_counts = frame["label"].value_counts().sort_index()
    missing_classes = [label for label in CLASS_LABELS if label not in class_counts]
    if missing_classes:
        raise ValueError(f"Dataset sınıfları eksik: {missing_classes}")

    return frame


def geometric_normalize(raw_features: np.ndarray) -> np.ndarray:
    landmarks = raw_features.reshape(-1, LANDMARK_COUNT, COORDINATE_COUNT).copy()

    # Translation invariance: bileği koordinat merkezine taşı.
    landmarks -= landmarks[:, 0:1, :]

    # Scale invariance: avuç uzunluğunu referans al.
    palm_vector = landmarks[:, 9, :]
    palm_scale = np.linalg.norm(palm_vector, axis=1)
    fallback_scale = np.max(np.linalg.norm(landmarks, axis=2), axis=1)
    palm_scale = np.where(palm_scale > 1e-6, palm_scale, fallback_scale)
    palm_scale = np.clip(palm_scale, 1e-6, None)
    landmarks /= palm_scale[:, None, None]

    # Rotation invariance: wrist -> middle MCP vektörünü yukarı hizala.
    palm_angle = np.arctan2(landmarks[:, 9, 1], landmarks[:, 9, 0])
    rotation_angle = (-np.pi / 2) - palm_angle
    cosine = np.cos(rotation_angle)
    sine = np.sin(rotation_angle)
    x_values = landmarks[:, :, 0].copy()
    y_values = landmarks[:, :, 1].copy()
    landmarks[:, :, 0] = (
        cosine[:, None] * x_values - sine[:, None] * y_values
    )
    landmarks[:, :, 1] = (
        sine[:, None] * x_values + cosine[:, None] * y_values
    )

    # Aykırı MediaPipe koordinatlarının eğitimi bozmasını sınırla.
    landmarks = np.clip(landmarks, -4.0, 4.0)
    return landmarks.reshape(-1, INPUT_SIZE).astype(np.float32)


def create_temporal_blocks(
    frame: pd.DataFrame,
    *,
    block_size: int,
    gap_seconds: float,
) -> pd.Series:
    groups = pd.Series(index=frame.index, dtype="object")

    for label in CLASS_LABELS:
        class_rows = frame[frame["label"] == label].sort_values("captured_at")
        block_index = 0
        block_item_count = 0
        previous_time: pd.Timestamp | None = None

        for row_index, row in class_rows.iterrows():
            current_time = row["captured_at"]
            starts_new_block = (
                previous_time is None
                or (current_time - previous_time).total_seconds() > gap_seconds
                or block_item_count >= block_size
            )
            if starts_new_block:
                block_index += 1
                block_item_count = 0

            groups.loc[row_index] = f"{label}-{block_index}"
            block_item_count += 1
            previous_time = current_time

    return groups


def split_by_temporal_blocks(
    frame: pd.DataFrame,
    config: TrainingConfig,
) -> dict[str, np.ndarray]:
    random_generator = np.random.default_rng(config.seed)
    groups = create_temporal_blocks(
        frame,
        block_size=config.temporal_block_size,
        gap_seconds=config.temporal_gap_seconds,
    )
    split_indices: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for label in CLASS_LABELS:
        label_indices = frame.index[frame["label"] == label].to_numpy()
        label_groups = groups.loc[label_indices]
        unique_groups = label_groups.unique().tolist()
        random_generator.shuffle(unique_groups)

        if len(unique_groups) < 3:
            raise ValueError(
                f"{label}. sınıf için en az 3 temporal blok gerekli; "
                f"bulunan: {len(unique_groups)}"
            )

        test_group_count = max(
            1, round(len(unique_groups) * config.test_ratio)
        )
        validation_group_count = max(
            1, round(len(unique_groups) * config.validation_ratio)
        )
        if test_group_count + validation_group_count >= len(unique_groups):
            validation_group_count = 1
            test_group_count = 1

        test_groups = set(unique_groups[:test_group_count])
        validation_groups = set(
            unique_groups[
                test_group_count : test_group_count + validation_group_count
            ]
        )

        for row_index in label_indices:
            group = groups.loc[row_index]
            if group in test_groups:
                split_indices["test"].append(int(row_index))
            elif group in validation_groups:
                split_indices["validation"].append(int(row_index))
            else:
                split_indices["train"].append(int(row_index))

    return {
        name: np.array(sorted(indices), dtype=np.int64)
        for name, indices in split_indices.items()
    }


def make_split_summary(
    frame: pd.DataFrame,
    splits: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, indices in splits.items():
        counts = frame.loc[indices, "label"].value_counts()
        for label in CLASS_LABELS:
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "sample_count": int(counts.get(label, 0)),
                }
            )
    return pd.DataFrame(rows)


def build_loaders(
    features: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    config: TrainingConfig,
) -> tuple[dict[str, DataLoader], np.ndarray, np.ndarray]:
    train_features = features[splits["train"]]
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    standardized = (features - feature_mean) / feature_std

    loaders: dict[str, DataLoader] = {}
    for split_name, indices in splits.items():
        dataset = LandmarkDataset(
            standardized[indices],
            labels[indices],
            augment=split_name == "train",
            noise_std=config.augmentation_noise_std,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split_name == "train",
            num_workers=0,
        )

    return (
        loaders,
        feature_mean.astype(np.float32),
        feature_std.astype(np.float32),
    )


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        sample_count += len(labels)
        all_probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probabilities = np.concatenate(all_probabilities)
    targets = np.concatenate(all_labels)
    predictions = probabilities.argmax(axis=1)
    accuracy = accuracy_score(targets, predictions)
    return total_loss / sample_count, accuracy, probabilities, targets


def train_model(
    loaders: dict[str, DataLoader],
    train_labels: np.ndarray,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[LeftHandMLP, dict[str, list[float]], int]:
    model = LeftHandMLP(
        dropout=config.dropout,
        class_count=len(CLASS_LABELS),
    ).to(device)
    class_counts = np.bincount(train_labels, minlength=len(CLASS_LABELS))
    class_weights = len(train_labels) / (
        len(CLASS_LABELS) * np.maximum(class_counts, 1)
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
        "learning_rate": [],
    }
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for features, labels in loaders["train"]:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(labels)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_count += len(labels)

        train_loss = train_loss_sum / train_count
        train_accuracy = train_correct / train_count
        validation_loss, validation_accuracy, _, _ = evaluate_loader(
            model,
            loaders["validation"],
            criterion,
            device,
        )
        scheduler.step(validation_loss)

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        improved = validation_loss < best_validation_loss - 1e-5
        if improved:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss {train_loss:.4f} acc {train_accuracy:.3f} | "
                f"val loss {validation_loss:.4f} acc {validation_accuracy:.3f}"
            )

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping: epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("En iyi model checkpointi oluşturulamadı.")

    model.load_state_dict(best_state)
    return model, history, best_epoch


def save_training_history(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["validation_loss"], label="Validation")
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["validation_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(
        targets,
        predictions,
        labels=np.arange(len(CLASS_LABELS)),
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Purples")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        xlabel="Predicted degree",
        ylabel="True degree",
        title="Test confusion matrix",
    )

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_confidence_plots(
    targets: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(0, 1, 21)
    axes[0].hist(
        confidence[correct],
        bins=bins,
        alpha=0.7,
        label="Correct",
        color="#6d28d9",
    )
    if (~correct).any():
        axes[0].hist(
            confidence[~correct],
            bins=bins,
            alpha=0.7,
            label="Incorrect",
            color="#e11d48",
        )
    axes[0].set(
        title="Softmax confidence distribution",
        xlabel="Maximum probability",
        ylabel="Sample count",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    calibration_bins = np.linspace(0, 1, 11)
    bin_centers: list[float] = []
    bin_accuracies: list[float] = []
    for lower, upper in zip(calibration_bins[:-1], calibration_bins[1:]):
        mask = (confidence >= lower) & (confidence < upper)
        if upper == 1.0:
            mask = (confidence >= lower) & (confidence <= upper)
        if mask.any():
            bin_centers.append(float(confidence[mask].mean()))
            bin_accuracies.append(float(correct[mask].mean()))

    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[1].plot(
        bin_centers,
        bin_accuracies,
        marker="o",
        color="#6d28d9",
        label="Model",
    )
    axes[1].set(
        title="Reliability diagram",
        xlabel="Mean confidence",
        ylabel="Observed accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def threshold_analysis(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    rows = []

    for threshold in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
        accepted = confidence >= threshold
        rows.append(
            {
                "threshold": threshold,
                "coverage": float(accepted.mean()),
                "accepted_samples": int(accepted.sum()),
                "accepted_accuracy": (
                    float((predictions[accepted] == targets[accepted]).mean())
                    if accepted.any()
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    set_reproducible_seed(config.seed)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_directory = args.output.resolve() / run_name
    run_directory.mkdir(parents=True, exist_ok=False)

    frame = load_dataset(args.csv.resolve())
    raw_features = frame[landmark_columns()].to_numpy(dtype=np.float32)
    features = geometric_normalize(raw_features)
    labels = frame["label"].map(
        {label: index for index, label in enumerate(CLASS_LABELS)}
    ).to_numpy(dtype=np.int64)
    splits = split_by_temporal_blocks(frame, config)
    split_summary = make_split_summary(frame, splits)

    loaders, feature_mean, feature_std = build_loaders(
        features,
        labels,
        splits,
        config,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Dataset: {len(frame)} örnek")
    print(frame["label"].value_counts().sort_index().to_string())
    print("\nSplit:")
    print(split_summary.pivot(index="label", columns="split", values="sample_count"))
    print(f"\nDevice: {device}")

    model, history, best_epoch = train_model(
        loaders,
        labels[splits["train"]],
        config,
        device,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, test_probabilities, test_targets = evaluate_loader(
        model,
        loaders["test"],
        criterion,
        device,
    )
    test_predictions = test_probabilities.argmax(axis=1)
    macro_f1 = f1_score(test_targets, test_predictions, average="macro")
    report = classification_report(
        test_targets,
        test_predictions,
        labels=np.arange(len(CLASS_LABELS)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    thresholds = threshold_analysis(test_targets, test_probabilities)

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "input_size": INPUT_SIZE,
        "hidden_sizes": [128, 64],
        "class_labels": CLASS_LABELS,
        "dropout": config.dropout,
        "feature_mean": torch.tensor(feature_mean),
        "feature_std": torch.tensor(feature_std),
        "preprocessing": {
            "center_landmark": 0,
            "scale_landmark": 9,
            "align_palm_to_negative_y": True,
            "clip_range": [-4.0, 4.0],
        },
    }
    torch.save(checkpoint, run_directory / "left_hand_model.pt")
    np.savez(
        run_directory / "preprocessing.npz",
        feature_mean=feature_mean,
        feature_std=feature_std,
        class_labels=np.array(CLASS_LABELS),
    )

    scripted_model = torch.jit.trace(
        model.eval(),
        torch.zeros(1, INPUT_SIZE, dtype=torch.float32),
    )
    scripted_model.save(str(run_directory / "left_hand_model_torchscript.pt"))

    metrics = {
        "dataset_size": len(frame),
        "class_counts": {
            str(key): int(value)
            for key, value in frame["label"].value_counts().sort_index().items()
        },
        "device": str(device),
        "best_epoch": best_epoch,
        "completed_epochs": len(history["train_loss"]),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_macro_f1": macro_f1,
        "config": asdict(config),
        "warning": (
            "Kişi/session metadatası bulunmadığı için test split temporal bloklarla "
            "oluşturuldu. Farklı kişilerdeki gerçek genelleme ayrıca ölçülmeli."
        ),
    }
    (run_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_directory / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(report).transpose().to_csv(
        run_directory / "classification_report.csv"
    )
    split_summary.to_csv(run_directory / "split_summary.csv", index=False)
    thresholds.to_csv(run_directory / "confidence_thresholds.csv", index=False)

    save_training_history(
        history,
        run_directory / "training_history.png",
    )
    save_confusion_matrix(
        test_targets,
        test_predictions,
        run_directory / "confusion_matrix.png",
    )
    save_confidence_plots(
        test_targets,
        test_probabilities,
        run_directory / "confidence_analysis.png",
    )

    print("\nEğitim tamamlandı")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(f"En iyi epoch: {best_epoch}")
    print(f"Çıktılar: {run_directory}")


if __name__ == "__main__":
    main()
