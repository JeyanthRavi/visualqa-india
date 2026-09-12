# VisualQA India

VisualQA India is a multimodal road-safety analysis application for road damage
and flood conditions. It combines YOLO-based road-damage detection with BLIP-2
visual question answering and produces a transparent 0–100 safety score through
a Gradio interface.

## Features

- Detects potholes and cracks with a trained YOLO model
- Analyses road and flood conditions with BLIP-2
- Displays detected damage on an annotated image
- Generates diagnostic observations for each safety category
- Calculates a deterministic 0–100 safety score
- Runs through an interactive Gradio interface
- Supports Kaggle GPU training and inference

## Architecture

```text
Input image
   ├── YOLO detector ──> road-damage detections
   └── BLIP-2 ─────────> visual diagnostic answers
                              │
                              ▼
                    Evidence-fusion layer
                              │
                              ▼
                 Safety score and Gradio report
```

## Models and datasets

- [BLIP-2 OPT-2.7B](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- [RDD2022](https://github.com/sekilab/RoadDamageDetector) for YOLO road-damage training
- [India Driving Dataset](https://inai.iiit.ac.in/domains/idd.html) for varied Indian road scenes
- [FloodNet](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021) for flood imagery

## Run on Kaggle

1. Create a Kaggle Notebook and enable a GPU accelerator.
2. Turn on Internet access so the pretrained BLIP-2 weights can be downloaded.
3. Add RDD2022 as a Kaggle input when training the YOLO detector.
4. Upload and run `kaggle_train_and_run_hybrid.ipynb`.

For inference with an existing YOLO checkpoint, attach `best.pt` as a Kaggle
input and run `kaggle_visualqa_india.ipynb`.

## Run locally

An NVIDIA GPU is recommended for BLIP-2.

```bash
git clone https://github.com/JeyanthRavi/visualqa-india.git
cd visualqa-india
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

To use a specific YOLO checkpoint:

```bash
VISUALQA_YOLO_MODEL=/path/to/best.pt python app.py
```

## Train the road-damage detector

```bash
python -m pip install -r requirements-training.txt
python train_rdd2022_yolo.py
```

Training outputs are saved under `visualqa_runs`. They include the trained
`best.pt` checkpoint, selected confidence threshold, evaluation metrics, plots,
and prediction results.

## Evaluation

The training pipeline reports:

- Precision
- Recall
- F1 score
- Accuracy and specificity at image level
- mAP50 and mAP50–95 for object detection
- Confusion matrix

## Tests

```bash
python -m pip install pytest
pytest -q
```

## Project structure

```text
visualqa-india/
├── app.py
├── train_rdd2022_yolo.py
├── kaggle_train_and_run_hybrid.ipynb
├── kaggle_visualqa_india.ipynb
├── requirements.txt
├── requirements-training.txt
├── data/
├── models/
├── tests/
└── visualqa/
    ├── analyzer.py
    ├── detector.py
    └── scoring.py
```

## Limitations

The safety score is a heuristic research output, not a replacement for a civil
engineering inspection or an emergency assessment. Results should be reviewed
with the original image before use.

## License

This project is licensed under the MIT License. Models and datasets retain their
original licenses and terms of use.
