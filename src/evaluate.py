"""
evaluate.py
-----------
Phase 4 (evaluation half): runs greedy-search inference over the held-out
test split, scores it with BLEU-1 and BLEU-4 (NLTK), and renders a figure
of N random test images with their generated vs. actual captions.

Deliverables produced here:
    outputs/bleu_scores.json
    outputs/predictions_test.csv         (every test image: generated + actual captions)
    outputs/sample_predictions.png       (N random test images, generated vs actual)

Usage:
    python src/evaluate.py
    python src/evaluate.py --limit 200   # score only a subset (faster smoke test)
"""

import argparse
import json
import os
import pickle
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    CLEAN_CAPTIONS_PATH,
    END_TOKEN,
    FEATURES_PATH,
    FINAL_MODEL_PATH,
    IMAGES_DIR,
    N_EVAL_VISUALS,
    OUTPUTS_DIR,
    RANDOM_SEED,
    SPLITS_PATH,
    START_TOKEN,
    TOKENIZER_PATH,
)
from inference import generate_caption  # noqa: E402


def _strip_control_tokens(caption: str) -> str:
    return " ".join(w for w in caption.split() if w not in (START_TOKEN, END_TOKEN))


def compute_bleu_scores(references_by_image: dict, hypotheses_by_image: dict):
    """
    references_by_image: {image_id: [ref_caption_1, ref_caption_2, ...]} (cleaned, tokenized as word lists)
    hypotheses_by_image: {image_id: hyp_caption} (cleaned, as a single string)
    Returns {"bleu1": float, "bleu4": float}
    """
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    references, hypotheses = [], []
    for image_id, hyp in hypotheses_by_image.items():
        refs = references_by_image.get(image_id)
        if not refs:
            continue
        references.append([r.split() for r in refs])
        hypotheses.append(hyp.split())

    smoothing = SmoothingFunction().method4
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    return {"bleu1": bleu1, "bleu4": bleu4, "n_images_scored": len(hypotheses)}


def run(limit: int = None):
    with open(CLEAN_CAPTIONS_PATH, "rb") as f:
        clean_captions = pickle.load(f)
    with open(TOKENIZER_PATH, "rb") as f:
        tok_bundle = pickle.load(f)
    with open(SPLITS_PATH, "rb") as f:
        splits = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)

    tokenizer = tok_bundle["tokenizer"]
    max_len = tok_bundle["max_len"]

    from tensorflow.keras.models import load_model
    if not os.path.isfile(FINAL_MODEL_PATH):
        raise FileNotFoundError(f"No trained model found at {FINAL_MODEL_PATH}. Run src/train.py first.")
    model = load_model(FINAL_MODEL_PATH)

    test_ids = [i for i in splits["test"] if i in features]
    if limit:
        test_ids = test_ids[:limit]
    print(f"Evaluating on {len(test_ids)} test images ...")

    references_by_image, hypotheses_by_image = {}, {}
    rows = [("image_id", "generated_caption", "actual_captions")]

    for image_id in test_ids:
        feature = features[image_id]
        hyp = generate_caption(model, feature, tokenizer, max_len)
        refs = [_strip_control_tokens(c) for c in clean_captions.get(image_id, [])]

        hypotheses_by_image[image_id] = hyp
        references_by_image[image_id] = refs
        rows.append((image_id, hyp, " | ".join(refs)))

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # --- Predictions CSV --------------------------------------------------
    import csv
    csv_path = os.path.join(OUTPUTS_DIR, "predictions_test.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved per-image predictions to {csv_path}")

    # --- BLEU scores --------------------------------------------------------
    scores = compute_bleu_scores(references_by_image, hypotheses_by_image)
    scores_path = os.path.join(OUTPUTS_DIR, "bleu_scores.json")
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"BLEU-1: {scores['bleu1']:.4f} | BLEU-4: {scores['bleu4']:.4f} (over {scores['n_images_scored']} images)")
    print(f"Saved BLEU scores to {scores_path}")

    # --- Visualization: N random test images, generated vs actual -------
    _render_sample_grid(test_ids, hypotheses_by_image, references_by_image)

    return scores


def _render_sample_grid(test_ids, hypotheses_by_image, references_by_image,
                         n=N_EVAL_VISUALS, seed=RANDOM_SEED):
    import matplotlib.pyplot as plt
    from PIL import Image

    sample_ids = random.Random(seed).sample(test_ids, min(n, len(test_ids)))
    if not sample_ids:
        print("No test images available to visualize.")
        return

    cols = 2
    rows = (len(sample_ids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, image_id in zip(axes, sample_ids):
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(IMAGES_DIR, image_id + ext)
            if os.path.isfile(candidate):
                img_path = candidate
                break
        if img_path:
            ax.imshow(Image.open(img_path))
        ax.axis("off")
        actual = references_by_image.get(image_id, ["?"])[0]
        ax.set_title(
            f"Generated: {hypotheses_by_image[image_id]}\nActual: {actual}",
            fontsize=9, wrap=True,
        )

    for ax in axes[len(sample_ids):]:
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "sample_predictions.png")
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample prediction grid to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N test images (smoke test)")
    args = parser.parse_args()
    run(limit=args.limit)
