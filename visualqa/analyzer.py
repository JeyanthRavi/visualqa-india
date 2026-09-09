"""BLIP-2 inference wrapper and end-to-end image analysis."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageOps

from .scoring import DIAGNOSTICS, score_answers


DEFAULT_MODEL_ID = "Salesforce/blip2-opt-2.7b"


def select_answer_tokens(output_ids, prompt_length: int, decoder_only: bool):
    """Remove prompt tokens returned by decoder-only language models."""
    if decoder_only and output_ids.shape[1] > prompt_length:
        return output_ids[:, prompt_length:]
    return output_ids


class VisualQAAnalyzer:
    """Lazily loads BLIP-2 so importing the project does not allocate a GPU."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.torch = torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = Blip2Processor.from_pretrained(model_id, use_fast=False)

        load_kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        if self.device.startswith("cuda"):
            load_kwargs["device_map"] = {"": 0}

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, **load_kwargs
        )
        if self.device == "cpu":
            self.model.to(self.device)
        self.model.eval()

    def ask(self, image: Image.Image, question: str) -> str:
        image = ImageOps.exif_transpose(image).convert("RGB")
        prompt = f"Question: {question} Answer:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {
            name: (
                tensor.to(self.device, dtype=self.dtype)
                if self.torch.is_floating_point(tensor)
                else tensor.to(self.device)
            )
            for name, tensor in inputs.items()
        }
        prompt_length = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=32,
                num_beams=3,
                do_sample=False,
            )
        output_ids = select_answer_tokens(
            output_ids,
            prompt_length=prompt_length,
            decoder_only=self.model.config.use_decoder_only_language_model,
        )
        return self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

    def analyze(self, image: Image.Image) -> dict:
        answers = {
            diagnostic.key: self.ask(image, diagnostic.question)
            for diagnostic in DIAGNOSTICS
        }
        report = score_answers(answers)
        report["answers"] = answers
        return report
