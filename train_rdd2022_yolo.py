"""Train and evaluate a YOLO road-damage detector on RDD2022.

Designed for a Kaggle GPU notebook. The validation split selects an image-level
confidence threshold; the test split is used once for final metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from ultralytics import YOLO


CLASS_NAMES = {
    0: "longitudinal crack",
    1: "transverse crack",
    2: "alligator crack",
    3: "other road damage",
    4: "pothole",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/kaggle/input/datasets/aliabdelmenam/rdd-2022/RDD_SPLIT"),
    )
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/visualqa_runs"))
    return parser.parse_args()


def validate_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            path = root / split / kind
            if not path.is_dir():
                raise FileNotFoundError(f"Required dataset directory not found: {path}")


def create_data_yaml(root: Path) -> Path:
    config = {
        "path": str(root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": CLASS_NAMES,
    }
    destination = Path("/kaggle/working/rdd2022.yaml")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False))
    return destination


def ground_truth_has_damage(image_path: str | Path, label_dir: Path) -> int:
    label_path = label_dir / f"{Path(image_path).stem}.txt"
    if not label_path.exists():
        return 0
    return int(any(line.strip() for line in label_path.read_text().splitlines()))


def collect_image_scores(model: YOLO, root: Path, split: str, imgsz: int) -> pd.DataFrame:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    rows = []
    predictions = model.predict(
        source=str(image_dir),
        imgsz=imgsz,
        conf=0.001,
        iou=0.7,
        device=0,
        stream=True,
        verbose=False,
    )
    for result in predictions:
        confidence = (
            float(result.boxes.conf.max().cpu())
            if result.boxes is not None and len(result.boxes)
            else 0.0
        )
        rows.append(
            {
                "image_path": result.path,
                "ground_truth": ground_truth_has_damage(result.path, label_dir),
                "max_detection_confidence": confidence,
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(validation: pd.DataFrame) -> float:
    candidates = []
    for threshold in np.arange(0.05, 0.81, 0.025):
        prediction = (validation["max_detection_confidence"] >= threshold).astype(int)
        positive_f1 = f1_score(validation["ground_truth"], prediction, zero_division=0)
        balanced = balanced_accuracy_score(validation["ground_truth"], prediction)
        candidates.append((float((positive_f1 + balanced) / 2), float(threshold)))
    return max(candidates)[1]


def print_image_metrics(test: pd.DataFrame, threshold: float) -> None:
    test["prediction"] = (test["max_detection_confidence"] >= threshold).astype(int)
    y_true = test["ground_truth"].to_numpy()
    y_pred = test["prediction"].to_numpy()
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0

    print("\nFinal image-level test metrics")
    print(f"Threshold:         {threshold:.3f}")
    print(f"Accuracy:          {accuracy_score(y_true, y_pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision:         {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:            {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Specificity:       {specificity:.4f}")
    print(f"Positive F1:       {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Macro F1:          {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"MCC:               {matthews_corrcoef(y_true, y_pred):.4f}")
    print("Confusion matrix:\n", matrix)
    print(classification_report(y_true, y_pred, target_names=["No damage", "Damage"], zero_division=0))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Enable a Kaggle GPU accelerator.")
    validate_dataset(args.dataset_root)
    data_yaml = create_data_yaml(args.dataset_root)

    detector = YOLO(args.model)
    detector.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=12,
        imgsz=args.imgsz,
        batch=-1,
        device=0,
        workers=4,
        optimizer="auto",
        amp=True,
        mosaic=0.7,
        mixup=0.0,
        close_mosaic=10,
        cache=False,
        pretrained=True,
        project=str(args.output),
        name="rdd2022_detector",
        exist_ok=True,
        plots=True,
        save=True,
        save_period=10,
    )

    best_path = Path(detector.trainer.best)
    best = YOLO(str(best_path))
    detection = best.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=-1,
        device=0,
        conf=0.001,
        iou=0.7,
        plots=True,
        save_json=True,
        project=str(args.output),
        name="rdd2022_test_evaluation",
        exist_ok=True,
    )
    print("\nObject-detection test metrics")
    print(f"Mean precision: {float(detection.box.mp):.4f}")
    print(f"Mean recall:    {float(detection.box.mr):.4f}")
    print(f"mAP50:          {float(detection.box.map50):.4f}")
    print(f"mAP50-95:       {float(detection.box.map):.4f}")

    validation = collect_image_scores(best, args.dataset_root, "val", args.imgsz)
    threshold = choose_threshold(validation)
    test = collect_image_scores(best, args.dataset_root, "test", args.imgsz)
    print_image_metrics(test, threshold)
    results_path = args.output / "rdd2022_image_level_test.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(results_path, index=False)
    print(f"Best model: {best_path}")
    print(f"Predictions: {results_path}")


if __name__ == "__main__":
    main()
