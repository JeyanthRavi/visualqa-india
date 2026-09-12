# Evaluation data

The pretrained BLIP-2 component accepts JPG and PNG images through Gradio. The
RDD2022 dataset is required only for training or evaluating the YOLO
road-damage detector; inference can use an existing `best.pt` checkpoint.

A meaningful evaluation requires a balanced set of Indian road and flood images
under `data/raw/` with labels recorded in `evaluation_template.csv`. Raw images
must remain outside Git unless their licence explicitly permits redistribution.

Suggested sources:

- **RDD2022 India subset** for potholes and road damage. It includes Pascal VOC
  bounding-box annotations.
- **India Driving Dataset (IDD)** for varied Indian road scenes and negative
  examples without damage.
- **FloodNet** for flood imagery. Note that it is aerial, so it does not perfectly
  match phone/dashcam uploads; add locally collected, consented street-level flood
  images if possible.

Performance claims should be based on at least 200–500 manually labelled images
covering day/night, rain/dry, urban/rural, paved/unpaved, and different camera
positions. Splitting by location or capture sequence reduces near-duplicate
leakage.
