"""
prepare_captions.py
--------------------
Phase 2 of the pipeline: load the raw Flickr8k captions, clean them,
wrap each with startseq/endseq, build a Keras Tokenizer (the vocabulary),
and split image IDs into train/val/test sets.

Produces (all under MODELS_DIR):
    clean_captions.pkl   -> {image_id: [caption_1, ..., caption_5]}
    tokenizer.pkl         -> fitted keras Tokenizer
    splits.pkl            -> {"train": [...], "val": [...], "test": [...]}

Run standalone:
    python src/prepare_captions.py
"""

import os
import pickle
import re
import string
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    CAPTIONS_FILE,
    CLEAN_CAPTIONS_PATH,
    END_TOKEN,
    MIN_WORD_FREQ,
    OOV_TOKEN,
    RANDOM_SEED,
    SPLITS_PATH,
    START_TOKEN,
    TEST_SPLIT,
    TOKENIZER_PATH,
    VAL_SPLIT,
)

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def load_raw_captions(captions_file: str = CAPTIONS_FILE) -> dict:
    """
    Reads a Flickr8k-style captions.txt with a header "image,caption" and
    returns {image_id: [caption_1, caption_2, ...]}. Tolerant of captions
    that themselves contain commas (joins everything after the first comma).
    """
    mapping = defaultdict(list)
    with open(captions_file, "r", encoding="utf-8") as f:
        header_skipped = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not header_skipped:
                header_skipped = True
                if line.lower().startswith("image,caption") or line.lower().startswith("image_name,comment"):
                    continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            image_name, caption = parts
            image_id = os.path.splitext(image_name.strip())[0]
            mapping[image_id].append(caption.strip())
    return dict(mapping)


def clean_caption(caption: str) -> str:
    """Lowercase, strip punctuation/digits, collapse whitespace, drop 1-letter tokens."""
    caption = caption.lower()
    caption = caption.translate(_PUNCT_TABLE)
    caption = re.sub(r"\d+", "", caption)
    words = [w for w in caption.split() if len(w) > 1 or w in ("a", "i")]
    return " ".join(words)


def build_clean_captions(raw_mapping: dict) -> dict:
    clean = {}
    for image_id, captions in raw_mapping.items():
        cleaned_list = []
        for c in captions:
            c_clean = clean_caption(c)
            if not c_clean:
                continue
            cleaned_list.append(f"{START_TOKEN} {c_clean} {END_TOKEN}")
        if cleaned_list:
            clean[image_id] = cleaned_list
    return clean


def build_tokenizer(clean_mapping: dict):
    """
    Fits a Keras Tokenizer over every cleaned caption. Words below
    MIN_WORD_FREQ are pruned before fitting, which keeps the softmax
    output layer (and therefore the model) a manageable size.
    """
    from tensorflow.keras.preprocessing.text import Tokenizer

    all_captions = [c for caps in clean_mapping.values() for c in caps]

    # First pass: word frequencies, to prune the long tail of rare words.
    freq = defaultdict(int)
    for c in all_captions:
        for w in c.split():
            freq[w] += 1
    vocab_words = {w for w, n in freq.items() if n >= MIN_WORD_FREQ}
    # Always keep the control tokens even if they're somehow under threshold.
    vocab_words.update({START_TOKEN, END_TOKEN})

    filtered_captions = []
    for c in all_captions:
        kept = [w for w in c.split() if w in vocab_words]
        filtered_captions.append(" ".join(kept))

    tokenizer = Tokenizer(oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(filtered_captions)
    return tokenizer


def max_caption_length(clean_mapping: dict) -> int:
    return max(len(c.split()) for caps in clean_mapping.values() for c in caps)


def train_val_test_split(image_ids: list):
    import random

    ids = list(image_ids)
    random.Random(RANDOM_SEED).shuffle(ids)
    n = len(ids)
    n_test = int(n * TEST_SPLIT)
    n_val = int(n * VAL_SPLIT)
    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def run():
    print(f"Loading raw captions from {CAPTIONS_FILE} ...")
    raw = load_raw_captions()
    print(f"  {len(raw)} images with captions found.")

    print("Cleaning captions and adding startseq/endseq ...")
    clean = build_clean_captions(raw)

    print("Building vocabulary (Tokenizer) ...")
    tokenizer = build_tokenizer(clean)
    vocab_size = len(tokenizer.word_index) + 1  # +1 for the 0 padding index
    max_len = max_caption_length(clean)
    print(f"  Vocabulary size (incl. padding idx): {vocab_size}")
    print(f"  Max caption length: {max_len} tokens")

    print("Splitting into train/val/test ...")
    splits = train_val_test_split(list(clean.keys()))
    for k, v in splits.items():
        print(f"  {k}: {len(v)} images")

    with open(CLEAN_CAPTIONS_PATH, "wb") as f:
        pickle.dump(clean, f)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump({"tokenizer": tokenizer, "vocab_size": vocab_size, "max_len": max_len}, f)
    with open(SPLITS_PATH, "wb") as f:
        pickle.dump(splits, f)

    print(f"\nSaved: {CLEAN_CAPTIONS_PATH}\n       {TOKENIZER_PATH}\n       {SPLITS_PATH}")
    return clean, tokenizer, vocab_size, max_len, splits


if __name__ == "__main__":
    run()
