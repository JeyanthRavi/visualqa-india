from PIL import Image

import app
from visualqa.scoring import DIAGNOSTICS, score_answers


class FakeAnalyzer:
    def analyze(self, image, risk_overrides=None):
        answers = {diagnostic.key: "No, none is visible." for diagnostic in DIAGNOSTICS}
        report = score_answers(answers, risk_overrides=risk_overrides)
        report["answers"] = answers
        return report

    def ask(self, image, question):
        return "No."


class FakeDetector:
    def detect(self, image):
        return {
            "risk": True,
            "evidence": "YOLO detected 1 defect: pothole (90%)",
            "detections": [{"label": "pothole", "confidence": 0.9}],
            "annotated_image": image,
            "model_path": "/tmp/best.pt",
            "confidence_threshold": 0.25,
        }


def test_gradio_analysis_fuses_detector_and_blip(monkeypatch) -> None:
    monkeypatch.setattr(app, "get_analyzer", lambda: FakeAnalyzer())
    monkeypatch.setattr(app, "get_detector", lambda: FakeDetector())
    image = Image.new("RGB", (16, 16), "gray")

    report, custom, annotated = app.analyze_image(image, "")

    assert "Heuristic safety score: 75 / 100" in report
    assert "specialist detector" in report
    assert "1 defect(s)" in report
    assert custom == ""
    assert annotated is image


def test_gradio_analysis_requires_an_image() -> None:
    assert app.analyze_image(None, "") == ("Please upload an image.", "", None)


def test_detector_failure_falls_back_to_blip(monkeypatch) -> None:
    monkeypatch.setattr(app, "get_analyzer", lambda: FakeAnalyzer())

    def fail_to_load_detector():
        raise RuntimeError("incompatible checkpoint")

    monkeypatch.setattr(app, "get_detector", fail_to_load_detector)
    image = Image.new("RGB", (16, 16), "gray")

    report, _, annotated = app.analyze_image(image, "")

    assert "BLIP-2 fallback active" in report
    assert "incompatible checkpoint" in report
    assert annotated is image
