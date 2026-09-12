"""Optional YOLO road-damage inference for the hybrid application."""

from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image, ImageOps


DEFAULT_MODEL_CANDIDATES = (
    Path("models/best.pt"),
    Path("/kaggle/working/visualqa_runs/best.pt"),
    Path("/kaggle/working/visualqa_runs/rdd2022_detector/weights/best.pt"),
)


def resolve_yolo_model_path(explicit: str | Path | None = None) -> Path | None:
    """Find a trained detector from an argument, environment, or known paths."""
    requested = explicit or os.getenv("VISUALQA_YOLO_MODEL")
    if requested:
        path = Path(requested).expanduser()
        return path.resolve() if path.is_file() else None

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        matches = sorted(kaggle_input.rglob("best.pt"))
        if matches:
            return matches[0].resolve()
    return None


def _load_saved_threshold(model_path: Path) -> float:
    configured = os.getenv("VISUALQA_YOLO_CONFIDENCE")
    if configured is not None:
        return float(configured)

    candidates = (
        model_path.with_name("detector_config.json"),
        model_path.parent.parent.parent / "detector_config.json",
    )
    for config_path in candidates:
        if config_path.is_file():
            data = json.loads(config_path.read_text())
            return float(data.get("confidence_threshold", 0.25))
    return 0.25


class RoadDamageDetector:
    """Run a trained RDD2022 YOLO checkpoint and return structured evidence."""

    def __init__(self, model_path: str | Path, confidence: float | None = None):
        from ultralytics import YOLO

        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {self.model_path}")
        self.confidence = (
            float(confidence)
            if confidence is not None
            else _load_saved_threshold(self.model_path)
        )
        self.model = YOLO(str(self.model_path))

    def detect(self, image: Image.Image) -> dict:
        import numpy as np
        import torch

        rgb_image = ImageOps.exif_transpose(image).convert("RGB")
        device = 0 if torch.cuda.is_available() else "cpu"
        result = self.model.predict(
            source=np.asarray(rgb_image),
            imgsz=768,
            conf=self.confidence,
            iou=0.7,
            device=device,
            verbose=False,
        )[0]

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append(
                    {
                        "class_id": class_id,
                        "label": str(result.names[class_id]),
                        "confidence": float(box.conf.item()),
                        "xyxy": [float(value) for value in box.xyxy[0].tolist()],
                    }
                )

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        if detections:
            summary = ", ".join(
                f"{item['label']} ({item['confidence']:.0%})"
                for item in detections[:5]
            )
            evidence = f"YOLO detected {len(detections)} defect(s): {summary}"
        else:
            evidence = (
                "YOLO found no road defect above the calibrated "
                f"{self.confidence:.2f} confidence threshold"
            )

        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[..., ::-1]
        return {
            "risk": bool(detections),
            "evidence": evidence,
            "detections": detections,
            "annotated_image": Image.fromarray(annotated_rgb),
            "model_path": str(self.model_path),
            "confidence_threshold": self.confidence,
        }
