# VisualQA India

VisualQA India is a hybrid research/demo application. A trained YOLO model
detects localized road defects, BLIP-2 answers broader questions about road and
flood scenes, and a transparent evidence-fusion layer converts the results into
a 0–100 heuristic safety score. Without a trained `best.pt`, the app clearly
falls back to BLIP-2-only analysis.

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
    ├→ trained YOLO → localized cracks/potholes + annotated image
    └→ BLIP-2 vision encoder + Q-Former + OPT-2.7B → scene answers
                     ↓
        evidence fusion (YOLO overrides only road-damage VQA)
                     ↓
       deterministic penalties → Gradio report
```

Model: [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b)

## Run on Kaggle (recommended)

You do **not** need a dataset to run the application. You only need one or more
JPG/PNG test images. A dataset is needed when you want to measure accuracy or
fine-tune a model.

1. Create a Kaggle Notebook.
2. In **Notebook options**, select a **GPU** accelerator (a T4 is suitable) and
   turn **Internet on** so Hugging Face can download the model.
3. Use `kaggle_train_and_run_hybrid.ipynb` to train YOLO and run the complete
   system, or `kaggle_visualqa_india.ipynb` to run with previously trained
   `best.pt` weights attached as a Kaggle input.
4. Run all cells. The final cell verifies the model load before starting Gradio;
   the initial download/loading can take several minutes.
5. The final cell starts Gradio and prints a public link.

If Kaggle reports CUDA out-of-memory, restart the session and run only this
notebook. Do not load another model in the same session.

## Run locally

BLIP-2 OPT-2.7B is large; an NVIDIA GPU is strongly recommended.

```bash
git clone https://github.com/JeyanthRavi/visualqa-india.git
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

BLIP-2 performs **zero-shot inference**. The separate YOLO component is trained
on RDD2022. Fine-tuning BLIP-2 remains a separate research task requiring
image/question/answer examples rather than YOLO bounding-box annotations.

## Train the RDD2022 road-damage detector

For stronger road-damage results, use the included YOLO training pipeline on a
Kaggle GPU. Add the RDD 2022 dataset, restart the runtime so BLIP-2 is not using
GPU memory, and run:

```bash
python -m pip install -r requirements-training.txt
python train_rdd2022_yolo.py
```

The script trains on `train`, selects an image-level confidence threshold using
`val`, and reports final object-detection and image-level metrics on `test`. It
saves a deployable `best.pt`, `detector_config.json`, plots, and a CSV under
`/kaggle/working/visualqa_runs`. The application discovers these files
automatically. A checkpoint attached elsewhere can be selected with:

```bash
VISUALQA_YOLO_MODEL=/path/to/best.pt python app.py
```

## Test the reasoning layer

The tests do not download BLIP-2:

```bash
python -m pip install pytest
pytest -q
```

## GitHub repository

The maintained repository is:

```text
https://github.com/JeyanthRavi/visualqa-india
```

To contribute from a branch:

```bash
git checkout -b codex/my-change
git add <changed-files>
git commit -m "Describe the change"
git push -u origin codex/my-change
```

Do not add model weights or raw datasets to the repository; they are ignored or
downloaded at runtime.

## Project structure

```text
visualqa-india/
├── app.py
├── kaggle_visualqa_india.ipynb
├── kaggle_train_and_run_hybrid.ipynb
├── requirements.txt
├── requirements-training.txt
├── train_rdd2022_yolo.py
├── data/
├── tests/
├── models/                 # optional local best.pt (ignored by Git)
└── visualqa/
    ├── analyzer.py
    ├── detector.py
    └── scoring.py
```

## Licence

Project code is MIT-licensed. Model and datasets retain their own licences and
terms.
