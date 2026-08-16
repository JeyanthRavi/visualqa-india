# Evaluation data

No dataset is required to run the pretrained BLIP-2 demo. Upload any JPG or PNG
through Gradio.

For a meaningful evaluation, create a balanced set of Indian road and flood
images under `data/raw/` and complete `evaluation_template.csv`. Keep the raw
images out of Git unless their licence explicitly permits redistribution.

Suggested sources:

- **RDD2022 India subset** for potholes and road damage. It includes Pascal VOC
  bounding-box annotations.
- **India Driving Dataset (IDD)** for varied Indian road scenes and negative
  examples without damage.
- **FloodNet** for flood imagery. Note that it is aerial, so it does not perfectly
  match phone/dashcam uploads; add locally collected, consented street-level flood
  images if possible.

Before claiming performance, manually label at least 200–500 images across
day/night, rain/dry, urban/rural, paved/unpaved, and different camera positions.
Split by location or capture sequence to avoid near-duplicate leakage.
