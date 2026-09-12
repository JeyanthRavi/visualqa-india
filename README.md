# VisualQA India

I built VisualQA India as a hybrid research and demonstration application for
analysing Indian road and flood scenes. A trained YOLO model detects localized
road defects, BLIP-2 answers broader visual questions, and a transparent
evidence-fusion layer converts the results into a 0–100 heuristic safety score.
Without a trained `best.pt`, the application falls back to BLIP-2-only analysis.

> **Important:** this is not a calibrated risk model, civil-engineering
> inspection, or emergency decision system. BLIP-2 can hallucinate and its model
> card says it has not been tested for real-world deployment. Results require
> manual review alongside the source image.

## Design and implementation

The application combines open-ended visual question answering with deterministic
safety rules. The reasoning layer is designed to prevent negated observations,
such as “no potholes are visible,” from being treated as detected hazards. Its
main implementation features are:

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

Inference requires a JPG or PNG image and pretrained model weights. BLIP-2
weights are downloaded automatically. Hybrid road-damage detection additionally
requires a trained YOLO `best.pt` checkpoint. RDD2022 is required for training
and evaluating that detector, but not for inference with an existing checkpoint.

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

## Datasets

The project uses complementary datasets because no single source covers every
supported condition:

1. **[RDD2022](https://github.com/sekilab/RoadDamageDetector)** — start with its
   India subset for potholes/cracks and annotated road damage.
2. **[India Driving Dataset (IDD)](https://inai.iiit.ac.in/domains/idd.html)** —
   use varied Indian road scenes, especially clean/normal negative examples.
3. **[FloodNet](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021)** —
   use for flood imagery, while noting that aerial images differ from typical
   street-level uploads.

An application-specific evaluation set should reflect realistic user photos. A
useful initial target is 200–500 manually reviewed images, balanced across the
five risk labels and urban/rural, day/night, rain/dry, paved/unpaved, and
phone/dashcam viewpoints. The [`data/`](data/) directory contains the label
template. Dataset licences must be checked before redistribution, and large
datasets should not be committed to GitHub.

BLIP-2 performs **zero-shot inference**. The separate YOLO component is trained
on RDD2022. Fine-tuning BLIP-2 remains a separate research task requiring
image/question/answer examples rather than YOLO bounding-box annotations.

## Train the RDD2022 road-damage detector

The included YOLO training pipeline produces the road-damage detector on a
Kaggle GPU. After attaching RDD2022, the following commands install the training
dependencies and start training:

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

## Repository policy

Model weights and raw datasets are excluded from version control. They are
downloaded at runtime or attached separately in Kaggle, subject to their
respective licences.

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
