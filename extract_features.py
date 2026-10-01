import sys
from pathlib import Path
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input
)
from tensorflow.keras.models import Model


ROOT = Path(__file__).resolve().parent.parent

IMAGES_DIR = ROOT / "data" / "Images"
MODELS_DIR = ROOT / "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


def build_model():

    print("Loading VGG16...")

    base_model = VGG16(
        weights="imagenet",
        include_top=True
    )

    model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer("fc2").output
    )

    print("VGG16 loaded.")
    print("Feature size: 4096")

    return model


def load_image(path):

    image = Image.open(path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image = np.asarray(image, dtype=np.float32)
    image = preprocess_input(image)

    return image


def extract_features():

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Images folder not found:\n{IMAGES_DIR}"
        )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_paths = sorted(
        [
            path
            for path in IMAGES_DIR.iterdir()
            if path.suffix.lower()
            in {".jpg", ".jpeg", ".png"}
        ]
    )

    if not image_paths:
        raise RuntimeError(
            "No images found in data/Images."
        )

    print()
    print("=" * 50)
    print("VISIONTALK FEATURE EXTRACTION")
    print("=" * 50)
    print(f"Images found: {len(image_paths)}")
    print(f"Image size: {IMAGE_SIZE}")
    print("Backbone: VGG16")
    print("Output layer: fc2")
    print("Feature size: 4096")
    print("=" * 50)
    print()

    model = build_model()

    features = {}

    total = len(image_paths)

    for start in range(0, total, BATCH_SIZE):

        batch_paths = image_paths[
            start:start + BATCH_SIZE
        ]

        batch_images = []

        valid_paths = []

        for path in batch_paths:

            try:
                image = load_image(path)

                batch_images.append(image)
                valid_paths.append(path)

            except Exception as e:

                print(
                    f"Skipping {path.name}: {e}"
                )

        if not batch_images:
            continue

        batch = np.array(
            batch_images,
            dtype=np.float32
        )

        batch_features = model.predict(
            batch,
            verbose=0
        )

        for path, feature in zip(
            valid_paths,
            batch_features
        ):

            image_id = path.stem

            features[image_id] = feature.astype(
                np.float32
            )

        processed = min(
            start + len(batch_paths),
            total
        )

        print(
            f"Processed {processed}/{total}"
        )

    output_file = MODELS_DIR / "features.pkl"

    with open(
        output_file,
        "wb"
    ) as f:

        pickle.dump(
            features,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print()
    print("=" * 50)
    print("FEATURE EXTRACTION COMPLETED")
    print("=" * 50)
    print(f"Images processed: {len(features)}")
    print("Feature dimension: 4096")
    print(f"Saved to: {output_file}")
    print("=" * 50)


if __name__ == "__main__":
    extract_features()