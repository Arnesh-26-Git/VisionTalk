"""
extract_features.py
--------------------
Phase 1 (encoder half): run every image through a pre-trained CNN
(VGG16 by default, ResNet50 / InceptionV3 also supported via config.py)
with its classification head removed, and cache the resulting feature
vector for every image to a single pickle file.

This is the expensive, one-time step — after this runs once, training
never has to touch VGG16 again, and never has to load raw images either.

Produces:
    models/features.pkl  -> {image_id: np.ndarray of shape (FEATURE_DIM,)}

Run standalone:
    python src/extract_features.py
    python src/extract_features.py --limit 200   # smoke-test on a subset
"""

import argparse
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CNN_BACKBONE, FEATURES_PATH, IMAGES_DIR, IMAGE_SIZE  # noqa: E402


def build_encoder():
    """
    Loads the chosen pre-trained CNN with its final classification layer
    stripped off, so the model's output is the penultimate feature vector
    instead of a 1000-way ImageNet softmax.
    """
    from tensorflow.keras.models import Model

    if CNN_BACKBONE == "vgg16":
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
        base = VGG16(weights="imagenet")
        # base.layers[-2] is the 4096-d fc2 layer, just before the softmax head.
        encoder = Model(inputs=base.inputs, outputs=base.layers[-2].output)
    elif CNN_BACKBONE == "resnet50":
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        base = ResNet50(weights="imagenet", include_top=True)
        encoder = Model(inputs=base.inputs, outputs=base.layers[-2].output)
    elif CNN_BACKBONE == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        base = InceptionV3(weights="imagenet", include_top=True)
        encoder = Model(inputs=base.inputs, outputs=base.layers[-2].output)
    else:
        raise ValueError(f"Unsupported CNN_BACKBONE: {CNN_BACKBONE}")

    return encoder, preprocess_input


def extract_all_features(images_dir: str = IMAGES_DIR, limit: int = None) -> dict:
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    encoder, preprocess_input = build_encoder()

    filenames = sorted(
        f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if limit:
        filenames = filenames[:limit]

    features = {}
    for fn in tqdm(filenames, desc=f"Extracting features ({CNN_BACKBONE})"):
        path = os.path.join(images_dir, fn)
        try:
            img = load_img(path, target_size=IMAGE_SIZE)
        except Exception as e:  # noqa: BLE001 — skip unreadable/corrupt files, keep going
            print(f"  ! Skipping {fn}: {e}")
            continue
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        vec = encoder.predict(arr, verbose=0)[0]
        image_id = os.path.splitext(fn)[0]
        features[image_id] = vec

    return features


def run(limit: int = None):
    if not os.path.isdir(IMAGES_DIR) or not os.listdir(IMAGES_DIR):
        print(
            f"No images found in {IMAGES_DIR}.\n"
            "Run `python src/download_dataset.py` first (see its output for "
            "manual instructions if automatic download fails)."
        )
        sys.exit(1)

    print(f"Backbone: {CNN_BACKBONE} | image size: {IMAGE_SIZE}")
    features = extract_all_features(limit=limit)
    print(f"Extracted features for {len(features)} images.")

    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(features, f)
    print(f"Saved to {FEATURES_PATH}")
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process first N images (smoke test)")
    args = parser.parse_args()
    run(limit=args.limit)
