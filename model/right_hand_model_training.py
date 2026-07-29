"""
ChordFlow sağ el classifier eğitimi.

Sınıflar:
    0 = Unknown
    1..5 = Dinamik uygulama sınıfları

Sol el eğitimindeki geometrik normalizasyon, temporal block split, MLP,
early stopping ve raporlama fonksiyonlarını yeniden kullanır.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn

import left_hand_model_training as training


CLASS_LABELS = [0, 1, 2, 3, 4, 5]
CLASS_NAMES = ["unknown", *[f"class_{label}" for label in CLASS_LABELS[1:]]]

# Ortak fonksiyonlar bu modül global değerlerini kullandığı için sağ el
# sınıf uzayını eğitim başlamadan önce tanımlıyoruz.
training.CLASS_LABELS = CLASS_LABELS
training.CLASS_NAMES = CLASS_NAMES


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="ChordFlow sağ el landmark modelini eğitir."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=project_root / "model" / "data" / "right" / "landmarks.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "model" / "training_outputs" / "right_hand",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = training.TrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    training.set_reproducible_seed(config.seed)

    run_directory = (
        args.output.resolve()
        / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )
    run_directory.mkdir(parents=True, exist_ok=False)

    frame = training.load_dataset(args.csv.resolve())
    raw_features = frame[training.landmark_columns()].to_numpy(dtype=np.float32)
    features = training.geometric_normalize(raw_features)
    labels = frame["label"].map(
        {label: index for index, label in enumerate(CLASS_LABELS)}
    ).to_numpy(dtype=np.int64)
    splits = training.split_by_temporal_blocks(frame, config)
    split_summary = training.make_split_summary(frame, splits)
    loaders, feature_mean, feature_std = training.build_loaders(
        features,
        labels,
        splits,
        config,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Dataset: {len(frame)} örnek")
    print(frame["label"].value_counts().sort_index().to_string())
    print("\nSplit:")
    print(
        split_summary.pivot(
            index="label",
            columns="split",
            values="sample_count",
        )
    )
    print(f"\nDevice: {device}")

    model, history, best_epoch = training.train_model(
        loaders,
        labels[splits["train"]],
        config,
        device,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, test_probabilities, test_targets = (
        training.evaluate_loader(
            model,
            loaders["test"],
            criterion,
            device,
        )
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
    thresholds = training.threshold_analysis(
        test_targets,
        test_probabilities,
    )

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "input_size": training.INPUT_SIZE,
        "hidden_sizes": [128, 64],
        "class_labels": CLASS_LABELS,
        "class_names": CLASS_NAMES,
        "hand": "right",
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
    torch.save(checkpoint, run_directory / "right_hand_model.pt")
    np.savez(
        run_directory / "preprocessing.npz",
        feature_mean=feature_mean,
        feature_std=feature_std,
        class_labels=np.array(CLASS_LABELS),
    )

    scripted_model = torch.jit.trace(
        model.eval(),
        torch.zeros(1, training.INPUT_SIZE, dtype=torch.float32),
    )
    scripted_model.save(
        str(run_directory / "right_hand_model_torchscript.pt")
    )

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
            "Kişi/session metadatası bulunmadığı için test split temporal "
            "bloklarla oluşturuldu."
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
    thresholds.to_csv(
        run_directory / "confidence_thresholds.csv",
        index=False,
    )

    test_rows = frame.loc[
        splits["test"],
        ["id", "label", "image_path", "captured_at"],
    ].reset_index(drop=True)
    test_rows["predicted_label"] = np.array(CLASS_LABELS)[test_predictions]
    test_rows["confidence"] = test_probabilities.max(axis=1)
    for class_index, class_label in enumerate(CLASS_LABELS):
        test_rows[f"probability_{class_label}"] = test_probabilities[
            :, class_index
        ]
    test_rows.to_csv(run_directory / "test_predictions.csv", index=False)

    training.save_training_history(
        history,
        run_directory / "training_history.png",
    )
    training.save_confusion_matrix(
        test_targets,
        test_predictions,
        run_directory / "confusion_matrix.png",
    )
    training.save_confidence_plots(
        test_targets,
        test_probabilities,
        run_directory / "confidence_analysis.png",
    )

    print("\nSağ el eğitimi tamamlandı")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(f"En iyi epoch: {best_epoch}")
    print(f"Çıktılar: {run_directory}")


if __name__ == "__main__":
    main()
