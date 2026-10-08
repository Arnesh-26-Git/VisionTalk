# VisionTalk — Evaluation Report

> **Fill in the bracketed sections after you actually run the pipeline.**
> This template deliberately contains no invented numbers — every metric
> below only becomes real after `python src/train.py` and
> `python src/evaluate.py` have been run against your dataset.

## 1. Project summary

VisionTalk generates natural-language captions for images using a CNN
(VGG16) encoder for visual feature extraction and an LSTM decoder for
language generation, combined in a Keras Functional API "merge model."
Built as an assistive-technology case study for visually impaired users.

## 2. Dataset

- Dataset used: [Flickr8k / Flickr30k]
- Number of images: [fill in — e.g. 8,091]
- Captions per image: 5
- Train / Validation / Test split: [fill in counts, e.g. 6,473 / 809 / 809]
- Vocabulary size after cleaning (min frequency = `MIN_WORD_FREQ`): [fill in]
- Max caption length (tokens, including startseq/endseq): [fill in]

## 3. Architecture

| Component | Detail |
|---|---|
| CNN backbone | [VGG16 / ResNet50 / InceptionV3] |
| Feature vector dimension | [4096 for VGG16, 2048 for ResNet50/InceptionV3] |
| Embedding dimension | [from config.py `EMBEDDING_DIM`] |
| LSTM units | [from config.py `LSTM_UNITS`] |
| Dropout rate | [from config.py `DROPOUT_RATE`] |
| Merge strategy | Element-wise Add of image branch + text branch |
| Total trainable parameters | [paste from `model.summary()`] |

See `outputs/model_architecture.png` for the rendered diagram.

## 4. Training

| Hyperparameter | Value |
|---|---|
| Epochs (requested / actually run) | [fill in — early stopping may cut this short] |
| Batch size | [fill in] |
| Optimizer | Adam |
| Learning rate | [fill in] |
| Loss function | Categorical cross-entropy |

Training curve: see the loss/val_loss plot generated in
`notebooks/VisionTalk.ipynb` (Phase 4 cell), or re-plot from
`models/history.pkl`.

Final training loss: [fill in from `models/run_config.json`]
Final validation loss: [fill in from `models/run_config.json`]

## 5. Evaluation

| Metric | Score |
|---|---|
| BLEU-1 | [fill in from `outputs/bleu_scores.json`] |
| BLEU-4 | [fill in from `outputs/bleu_scores.json`] |
| Images scored | [fill in] |

BLEU-1 measures single-word overlap with the reference captions; BLEU-4
requires matching 4-word sequences and is expected to be substantially
lower — that gap is normal, not a defect.

### Sample predictions

See `outputs/sample_predictions.png` for 10 random test images with their
generated caption alongside a human-written reference caption. The full
per-image table (every test image, not just the sample) is in
`outputs/predictions_test.csv`.

## 6. Observations / known limitations

- Greedy search picks the single best next word at each step, which can
  occasionally lock in a suboptimal caption early on; beam search (not yet
  implemented here — `BEAM_WIDTH` in `config.py` is reserved for it) would
  typically improve quality at the cost of more compute per caption.
- Model quality scales with epochs trained and dataset size — Flickr8k
  (~8k images) is small by modern standards, so expect noticeably more
  generic/repetitive captions than a model trained on Flickr30k or MS COCO.
- BLEU rewards n-gram overlap with the specific reference captions
  collected for this dataset; a caption can be accurate and fluent while
  still scoring lower if it's phrased differently from all 5 references.

## 7. How to reproduce

```bash
python src/download_dataset.py
python src/prepare_captions.py
python src/extract_features.py
python src/train.py
python src/evaluate.py
```

See `README.md` for full instructions, configurable hyperparameters, and
troubleshooting.
