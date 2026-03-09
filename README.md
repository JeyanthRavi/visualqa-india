# VisualQA India 🇮🇳
### Multimodal Road Safety & Flood Severity Analyzer

A Vision-Language AI system that analyzes road conditions and flood severity directly from images using natural language reasoning.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow) ![Gradio](https://img.shields.io/badge/Gradio-4.0-orange)

---

## Overview

VisualQA India combines a pretrained Vision-Language Model (BLIP-2) with a custom keyword reasoning layer to automatically assess road safety and flood risk from a single image — no manual inspection required.

Given any road or area image, the system:
- Runs 5 targeted diagnostic questions through the VQA pipeline
- Parses model answers for risk indicators
- Computes a **Safety Score (0–100)**
- Classifies severity as Safe / Moderate / High Risk
- Accepts free-form natural language questions from the user

---

## Architecture

```
Image Input
    │
    ▼
ViT Vision Encoder (CLIP-style)
    │
    ▼
Q-Former (Querying Transformer)
    │
    ▼
OPT-2.7B Language Model  ◄──── Text Question
    │
    ▼
Natural Language Answer
    │
    ▼
Keyword Reasoning Layer
    │
    ▼
Safety Score + Severity Label
```

**Model:** [`Salesforce/blip2-opt-2.7b`](https://huggingface.co/Salesforce/blip2-opt-2.7b)

---

## Results

The model correctly identifies:
- Potholes and surface cracks → High Risk flag
- Waterlogging and flood presence → High Risk flag
- Construction obstructions → Moderate Risk flag
- Clear, well-maintained roads → Safe classification

---

## Getting Started

### Run on Google Colab (recommended)
Open `VisualQA_India.ipynb` in Google Colab with a T4 GPU runtime.

### Run locally
```bash
pip install transformers accelerate pillow gradio torch
jupyter notebook VisualQA_India.ipynb
```

> **Note:** Requires ~8GB VRAM. Use `torch_dtype=torch.float16` for memory efficiency.

---

## Usage

1. Upload any road, street, or flood area image
2. The system automatically runs 5 diagnostic questions
3. View the Safety Score and risk breakdown
4. Optionally ask your own question about the image

---

## Stack

| Component | Tool |
|---|---|
| Vision-Language Model | BLIP-2 (Salesforce) |
| Framework | PyTorch + HuggingFace Transformers |
| UI | Gradio |
| Runtime | Google Colab T4 GPU |

---

## Motivation

India faces significant challenges in road infrastructure monitoring and flood response. Manual inspection is slow, expensive, and inconsistent. This system explores how multimodal AI can automate visual assessment at scale — with direct applications in disaster response, smart city infrastructure, and public safety.

---

## License
MIT
