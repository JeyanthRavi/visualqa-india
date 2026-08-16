"""Gradio application for VisualQA India."""

from __future__ import annotations

import os
from functools import lru_cache

import gradio as gr
from PIL import Image

from visualqa import DIAGNOSTICS, VisualQAAnalyzer


@lru_cache(maxsize=1)
def get_analyzer() -> VisualQAAnalyzer:
    return VisualQAAnalyzer()


def _render_report(report: dict) -> str:
    icon = {"green": "🟢", "orange": "🟠", "red": "🔴"}[report["color"]]
    findings = "\n".join(
        f"- **{item['label']}**: −{item['penalty']} points "
        f"(evidence: `{item['evidence']}`)"
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

    return f"""## {icon} {report['severity']}

# Heuristic safety score: {report['score']} / 100

### Findings

{findings}{uncertain}

### Diagnostic answers

{chr(10).join(qa_lines)}

---
*Research/demo output only. This score is a transparent keyword heuristic, not a calibrated probability or an engineering safety inspection.*
"""


def analyze_image(image: Image.Image | None, custom_question: str):
    if image is None:
        return "Please upload an image.", ""
    analyzer = get_analyzer()
    report = analyzer.analyze(image)
    custom_answer = ""
    if custom_question.strip():
        answer = analyzer.ask(image, custom_question.strip())
        custom_answer = f"### Custom question\n\n**Q:** {custom_question.strip()}  \n**A:** {answer}"
    return _render_report(report), custom_answer


with gr.Blocks(title="VisualQA India", theme=gr.themes.Soft()) as demo:
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

    analyze_button.click(
        analyze_image,
        inputs=[image_input, custom_question],
        outputs=[report_output, custom_output],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true"
    )
