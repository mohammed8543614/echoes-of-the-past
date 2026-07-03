# 𓂀 Echoes of the Past

**An end-to-end pipeline for detecting and translating ancient Egyptian hieroglyphs from real stone inscriptions.**

Graduation Project — AI & Robotics, Al Balqa Applied University (2026)

---

## Overview

Echoes of the Past takes a photo of a hieroglyphic inscription — carved into a temple wall, a stele, whatever — and turns it into readable text. It detects individual Gardiner-classified glyphs, works out the correct reading order (including cartouches), and translates the sequence into natural language.

The pipeline has two core models sitting back to back:

- **Detection** — a YOLO11m model trained on ~55,000 images across 803 Gardiner sign classes, reaching **mAP50 of 0.824**
- **Translation** — a fine-tuned M2M100 transformer (`mattiadc/hiero-transformer`) that converts glyph-code sequences into translated text

A Streamlit interface ties it together into something you can actually click through.

## Pipeline Stages

1. **Data preparation** — building and cleaning a real-photograph dataset of hieroglyphs (moved away from an early cartoon/synthetic dataset — more on that below)
2. **Detection training** — YOLO11m trained on 803 classes at 640px, 100 epochs
3. **Glyph localization** — running inference to get bounding boxes + Gardiner codes for every glyph in an image
4. **Reading-order reconstruction** — sorting detected glyphs into the correct reading sequence, including detecting and unpacking     **cartouches** (royal name boxes, marked by the `V9` sign) as nested sub-sequences 
5. **Sequence scoring** — evaluating candidate glyph orderings to pick the most linguistically coherent one
6. **Translation** — feeding the ordered Gardiner-code sequence into the fine-tuned M2M100 transformer
7. **Presentation** — Streamlit UI for uploading an image and viewing detection + translation results

## Key Technical Decisions

A few choices in this project were deliberate and worth calling out, since they weren't the "default" option:

- **No horizontal flip augmentation (`fliplr=0.0`)** — most detection pipelines flip images for augmentation, but hieroglyphs are directional signs. Flipping a glyph changes what it means (Egyptians read *toward* the faces of the glyphs), so flip augmentation would have taught the model incorrect associations. Turned it off entirely.
- **Real stone-carving images over synthetic data** — the project initially trained on a merged cartoon + Roboflow dataset (~8,800 images, 96 classes). This was scrapped in favor of a dataset built from real inscription photography (803 classes), which generalizes far better to actual archaeological images.
- **Confidence threshold of 0.25** — tuned to balance catching faint/worn carvings against false positives on stone texture noise.
- **Pre-trained transformer over training from scratch** — earlier attempts (T5-base, custom M2M100 training) topped out around BLEU ~0.8–1.0. Switching to the pre-trained `mattiadc/hiero-transformer` jumped translation quality to **BLEU ~36**, since it already carried linguistic knowledge of Middle Egyptian this project's dataset alone couldn't provide.
- **Gardiner codes as atomic tokens** — glyphs are represented and processed as their standard Gardiner sign-list codes rather than raw pixel classes, keeping detection output directly compatible with Egyptological convention and the translation model's expected input.

## Results

| Component | Metric | Score |
|---|---|---|
| YOLO11m detection | mAP50 | 0.824 |
| Translation (M2M100 fine-tuned) | BLEU | ~36 |
| Detection classes | Gardiner sign classes | 803 |
| Training images | — | ~55,000 |

## Tech Stack

- **Detection:** YOLO11m (Ultralytics)
- **Translation:** M2M100 (Hugging Face Transformers), fine-tuned on the TLA corpus (`phiwi/bbaw_egyptian`, ~101K rows)
- **Dictionary lookup:** custom 918-entry Gardiner-to-English dataset
- **UI:** Streamlit
- **Training environment:** Kaggle / Google Colab (GPU)

## Repository Structure

```
echoes-of-the-past/
├── notebooks/
│   ├── 01_yolo_training.ipynb        # YOLO11m training on 803 classes
│   └── 02_inference_pipeline.ipynb   # detection + cartouche sorting + translation
├── src/
│   ├── glyph_sorting.py              # reading-order + cartouche detection logic
│   ├── translation.py                # M2M100 wrapper + sequence scoring
│   └── inference.py                  # end-to-end pipeline
├── streamlit_app/
│   └── app.py                        # UI
├── results/
│   ├── training_metrics.png          # mAP / loss curves
│   └── sample_predictions/           # example inputs & outputs
├── docs/
│   └── pipeline_architecture.png     # visual pipeline diagram
├── requirements.txt
└── README.md
```

## Setup & Usage

> Note: trained model weights are not included in this repo due to file size. Reach out if you'd like access for evaluation purposes.

```bash
# clone the repo
git clone https://github.com/YOUR_USERNAME/echoes-of-the-past.git
cd echoes-of-the-past

# install dependencies
pip install -r requirements.txt

# run the Streamlit app (requires model weights in place)
streamlit run streamlit_app/app.py
```

## Team

Built by **Mohammed**, **Batool**, and **Diana** as a graduation project for the AI & Robotics program at Al Balqa Applied University.

## Future Improvements

- Expand training data to cover rarer Gardiner sign classes with better representation
- Improve cartouche detection robustness on partially occluded/eroded inscriptions
- Multi-line inscription layout handling beyond simple row/cartouche logic

---

*This project is a research/academic pipeline exploring computer vision and NLP applications in Egyptology.*