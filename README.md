# VisualQA India

VisualQA India is a research/demo application that uses BLIP-2 to answer five
targeted questions about a road or flood image, then converts those generated
answers into a transparent 0–100 heuristic safety score.

> **Important:** this is not a calibrated risk model, civil-engineering
> inspection, or emergency decision system. BLIP-2 can hallucinate and its model
> card says it has not been tested for real-world deployment. Always review the
> image and answers manually.

## What was fixed

The original notebook was valid-looking but untested. In particular, a sentence
such as “no potholes are visible” still triggered the word `pothole`, and one
answer could receive both a high- and medium-risk deduction. This version:

- asks consistent harmful-condition yes/no questions;
- handles common negations and whole terms (`unsafe` no longer matches `safe`);
- applies at most one fixed penalty per diagnostic category;
- loads the large model lazily and keeps token tensors in an integer dtype;
- separates the model, scoring logic, Gradio UI, and tests;
- labels the output honestly as a heuristic score.

## Architecture

```text
uploaded image
    → BLIP-2 vision encoder + Q-Former + OPT-2.7B
    → five short diagnostic answers
    → deterministic category rules and penalties
    → Gradio report + optional custom VQA answer
```

Model: [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b)

## Run on Kaggle (recommended)

You do **not** need a dataset to run the application. You only need one or more
JPG/PNG test images. A dataset is needed when you want to measure accuracy or
fine-tune a model.

1. Create a Kaggle Notebook.
2. In **Notebook options**, select a **GPU** accelerator (a T4 is suitable) and
   turn **Internet on** so Hugging Face can download the model.
3. Upload this repository as a Kaggle Notebook, or open
   `kaggle_visualqa_india.ipynb` after replacing the GitHub URL in its setup cell.
4. Run all cells. Initial model download/loading can take several minutes.
5. The final cell starts Gradio and prints a public link.

If Kaggle reports CUDA out-of-memory, restart the session and run only this
notebook. Do not load another model in the same session.

## Run locally

BLIP-2 OPT-2.7B is large; an NVIDIA GPU is strongly recommended.

```bash
git clone https://github.com/YOUR_USERNAME/visualqa-india.git
cd visualqa-india
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

To create a temporary public Gradio URL:

```bash
GRADIO_SHARE=true python app.py
```

## Which dataset should you use?

Use a combination rather than a single dataset:

1. **[RDD2022](https://github.com/sekilab/RoadDamageDetector)** — start with its
   India subset for potholes/cracks and annotated road damage.
2. **[India Driving Dataset (IDD)](https://inai.iiit.ac.in/domains/idd.html)** —
   use varied Indian road scenes, especially clean/normal negative examples.
3. **[FloodNet](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021)** —
   use for flood imagery, while noting that aerial images differ from typical
   street-level uploads.

For this exact application, build a small evaluation set that matches real user
photos. Aim initially for 200–500 manually reviewed images, balanced across the
five risk labels and across urban/rural, day/night, rain/dry, paved/unpaved, and
phone/dashcam viewpoints. See [`data/`](data/) for the label template. Check each
dataset's licence before redistribution; do not commit large datasets to GitHub.

The current app performs **zero-shot inference**. It does not train on those
datasets. Fine-tuning is a separate project and requires converting annotations
to image/question/answer examples and evaluating on a location-separated test
set.

## Test the reasoning layer

The tests do not download BLIP-2:

```bash
python -m pip install pytest
pytest -q
```

## Upload to GitHub

Create an empty GitHub repository named `visualqa-india`, then run from this
folder:

```bash
git init
git add .
git commit -m "Build tested VisualQA India demo"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/visualqa-india.git
git push -u origin main
```

Do not add model weights or raw datasets to the repository; they are ignored or
downloaded at runtime.

## Project structure

```text
visualqa-india/
├── app.py
├── kaggle_visualqa_india.ipynb
├── requirements.txt
├── data/
├── tests/
└── visualqa/
    ├── analyzer.py
    └── scoring.py
```

## Licence

Project code is MIT-licensed. Model and datasets retain their own licences and
terms.
