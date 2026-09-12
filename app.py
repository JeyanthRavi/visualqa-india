"""Gradio application for VisualQA India."""

from __future__ import annotations

import os
from functools import lru_cache

import gradio as gr
from PIL import Image

from visualqa import (
    DIAGNOSTICS,
    RoadDamageDetector,
    VisualQAAnalyzer,
    resolve_yolo_model_path,
)


@lru_cache(maxsize=1)
def get_analyzer() -> VisualQAAnalyzer:
    return VisualQAAnalyzer()


@lru_cache(maxsize=1)
def get_detector() -> RoadDamageDetector | None:
    model_path = resolve_yolo_model_path()
    return RoadDamageDetector(model_path) if model_path else None


def _render_report(report: dict) -> str:
    icon = {"green": "🟢", "orange": "🟠", "red": "🔴"}[report["color"]]
    findings = "\n".join(
        f"- **{item['label']}**: −{item['penalty']} points "
        f"via **{item['source']}** (evidence: `{item['evidence']}`)"
        for item in report["findings"]
    ) or "- No explicit risk condition was detected in the generated answers."

    qa_lines = []
    for diagnostic in DIAGNOSTICS:
        qa_lines.append(
            f"**{diagnostic.label}**  \n{report['answers'][diagnostic.key]}"
        )

    uncertain = ""
    if report["unknown"]:
        uncertain = (
            "\n\n**Unclear answers:** " + ", ".join(report["unknown"])
            + ". Review these manually."
        )

    detector = report.get("road_detector")
    detector_error = report.get("road_detector_error")
    if detector:
        detector_text = (
            f"**Road detector:** active — {len(detector['detections'])} defect(s), "
            f"threshold {detector['confidence_threshold']:.2f}"
        )
    elif detector_error:
        detector_text = (
            "**Road detector:** failed to load or run; BLIP-2 fallback active. "
            f"Error: `{detector_error}`"
        )
    else:
        detector_text = (
            "**Road detector:** unavailable — using BLIP-2 for road damage. "
            "Attach a trained `best.pt` or set `VISUALQA_YOLO_MODEL`."
        )

    return f"""## {icon} {report['severity']}

# Heuristic safety score: {report['score']} / 100

{detector_text}

### Findings

{findings}{uncertain}

### Diagnostic answers

{chr(10).join(qa_lines)}

---
*Research/demo output only. This score is a transparent keyword heuristic, not a calibrated probability or an engineering safety inspection.*
"""


def analyze_image(image: Image.Image | None, custom_question: str):
    if image is None:
        return "Please upload an image.", "", None

    analyzer = get_analyzer()
    detector_error = None
    try:
        detector = get_detector()
        road_result = detector.detect(image) if detector else None
    except Exception as error:
        road_result = None
        detector_error = f"{type(error).__name__}: {error}"
    overrides = None
    if road_result:
        overrides = {
            "road_damage": (road_result["risk"], road_result["evidence"]),
        }

    report = analyzer.analyze(image, risk_overrides=overrides)
    report["road_detector"] = road_result
    report["road_detector_error"] = detector_error
    custom_answer = ""
    if custom_question and custom_question.strip():
        answer = analyzer.ask(image, custom_question.strip())
        custom_answer = f"### Custom question\n\n**Q:** {custom_question.strip()}  \n**A:** {answer}"
    annotated = road_result["annotated_image"] if road_result else image
    return _render_report(report), custom_answer, annotated


with gr.Blocks(title="VisualQA India") as demo:
    gr.Markdown(
        "# 🛣️ VisualQA India\n"
        "Upload a road or flood image for BLIP-2 observations and a transparent "
        "rule-based risk score. The first run downloads and loads the model."
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Road or flood image")
            custom_question = gr.Textbox(
                label="Optional custom question",
                placeholder="Example: Is the water above the vehicle tyres?",
            )
            analyze_button = gr.Button("Analyze", variant="primary")
        with gr.Column():
            report_output = gr.Markdown()
            custom_output = gr.Markdown()
            annotated_output = gr.Image(type="pil", label="YOLO road-damage detections")

    analyze_button.click(
        analyze_image,
        inputs=[image_input, custom_question],
        outputs=[report_output, custom_output, annotated_output],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true"
    )
