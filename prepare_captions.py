import sys
from pathlib import Path
import pickle
import re
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "Images"
MODELS_DIR = ROOT / "models"

SEED = 42
TRAIN_SPLIT = 0.8
MAX_VOCAB_SIZE = 10000


def clean_caption(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_caption_line(line):
    line = line.strip()

    if not line:
        return None

    # Flickr8k.token.txt format:
    # image.jpg#0<TAB>caption
    if "\t" in line:
        image_name, caption = line.split("\t", 1)
        return image_name, caption

    # captions.txt format:
    # image.jpg,caption
    if "," in line:
        image_name, caption = line.split(",", 1)
        return image_name, caption

    return None


def load_captions():

    possible_files = [
        DATA_DIR / "captions.txt",
        DATA_DIR / "Flickr8k.token.txt",
        ROOT / "captions.txt",
        ROOT / "Flickr8k.token.txt",
    ]

    captions_file = None

    for file in possible_files:
        if file.exists():
            captions_file = file
            break

    if captions_file is None:
        raise FileNotFoundError(
            "Caption file not found.\n"
            "Put Flickr8k.token.txt or captions.txt inside data/"
        )

    print(f"Using captions: {captions_file}")

    mapping = {}

    with open(
        captions_file,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:

            result = parse_caption_line(line)

            if result is None:
                continue

            image_name, caption = result

            image_name = image_name.split("#")[0].strip()

            image_id = Path(image_name).stem

            caption = clean_caption(caption)

            if not caption:
                continue

            caption = "startseq " + caption + " endseq"

            mapping.setdefault(
                image_id,
                []
            ).append(caption)

    return mapping


def build_tokenizer(captions):

    counter = Counter()

    for caption in captions:
        counter.update(caption.split())

    word_index = {
        "<pad>": 0,
        "<unk>": 1,
        "startseq": 2,
        "endseq": 3
    }

    words = [
        word
        for word, count in counter.most_common()
        if word not in {
            "startseq",
            "endseq"
        }
    ]

    words = words[:MAX_VOCAB_SIZE - len(word_index)]

    for word in words:
        if word not in word_index:
            word_index[word] = len(word_index)

    index_word = {
        value: key
        for key, value in word_index.items()
    }

    return {
        "word_index": word_index,
        "index_word": index_word
    }


def main():

    print("Preparing Flickr8k captions...")
    print()

    if not IMAGES_DIR.exists():

        raise FileNotFoundError(
            f"Images folder not found:\n{IMAGES_DIR}"
        )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    mapping = load_captions()

    image_ids = sorted(
        p.stem
        for p in IMAGES_DIR.iterdir()
        if p.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png"
        }
    )

    print(f"Images found: {len(image_ids)}")
    print(f"Caption entries found: {len(mapping)}")

    mapping = {
        image_id: mapping[image_id]
        for image_id in image_ids
        if image_id in mapping
    }

    if not mapping:

        raise RuntimeError(
            "No matching image-caption pairs found."
        )

    print(
        f"Matched images: {len(mapping)}"
    )

    ids = np.array(
        sorted(mapping.keys())
    )

    rng = np.random.default_rng(SEED)

    rng.shuffle(ids)

    n_train = int(
        len(ids) * TRAIN_SPLIT
    )

    remaining = ids[n_train:]

    n_val = len(remaining) // 2

    train_ids = ids[
        :n_train
    ].tolist()

    val_ids = remaining[
        :n_val
    ].tolist()

    test_ids = remaining[
        n_val:
    ].tolist()

    train_captions = []

    for image_id in train_ids:

        train_captions.extend(
            mapping[image_id]
        )

    tokenizer = build_tokenizer(
        train_captions
    )

    all_captions = []

    for captions in mapping.values():

        all_captions.extend(
            captions
        )

    max_length = max(
        len(caption.split())
        for caption in all_captions
    )

    vocab_size = len(
        tokenizer["word_index"]
    )

    with open(
        MODELS_DIR / "clean_captions.pkl",
        "wb"
    ) as f:

        pickle.dump(
            mapping,
            f
        )

    with open(
        MODELS_DIR / "splits.pkl",
        "wb"
    ) as f:

        pickle.dump(
            {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids
            },
            f
        )

    with open(
        MODELS_DIR / "tokenizer.pkl",
        "wb"
    ) as f:

        pickle.dump(
            tokenizer,
            f
        )

    with open(
        MODELS_DIR / "text_config.pkl",
        "wb"
    ) as f:

        pickle.dump(
            {
                "vocab_size": vocab_size,
                "max_length": max_length
            },
            f
        )

    print()
    print("=" * 45)
    print("CAPTION PREPARATION COMPLETED")
    print("=" * 45)
    print(
        f"Images: {len(mapping)}"
    )
    print(
        f"Train: {len(train_ids)}"
    )
    print(
        f"Validation: {len(val_ids)}"
    )
    print(
        f"Test: {len(test_ids)}"
    )
    print(
        f"Vocabulary size: {vocab_size}"
    )
    print(
        f"Maximum caption length: {max_length}"
    )
    print()
    print(
        f"Saved files to: {MODELS_DIR}"
    )


if __name__ == "__main__":
    main()