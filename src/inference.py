"""
inference.py
-------------
Turns a trained VisionTalk model + a single image into a natural-language
caption.

The module contains two main operations:

    extract_single_image_feature(path)
        Runs the same CNN encoder used during training on one new image.

    generate_caption(...)
        Generates one word at a time using greedy decoding.

This module is imported by both evaluate.py and the Streamlit app.
"""

import os
import pickle
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

sys.path.append(
    os.path.join(
        PROJECT_ROOT,
        "src"
    )
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from config import (  # noqa: E402
    CNN_BACKBONE,
    END_TOKEN,
    FINAL_MODEL_PATH,
    IMAGE_SIZE,
    MAX_CAPTION_LEN_CAP,
    START_TOKEN,
    TOKENIZER_PATH,
    VGG16_WEIGHTS_PATH,
)


_encoder_cache = {}


# ---------------------------------------------------------------------------
# CNN encoder
# ---------------------------------------------------------------------------
def _get_encoder():
    """
    Lazily builds and caches the same feature extractor used during training.
    """

    if "encoder" in _encoder_cache:
        return (
            _encoder_cache["encoder"],
            _encoder_cache["preprocess"],
        )

    from tensorflow.keras.models import Model

    if CNN_BACKBONE == "vgg16":

        from tensorflow.keras.applications.vgg16 import (
            VGG16,
            preprocess_input,
        )

        # Use local VGG16 weights when available.
        # Otherwise use the standard ImageNet weights.
        if os.path.isfile(VGG16_WEIGHTS_PATH):

            base = VGG16(
                weights=VGG16_WEIGHTS_PATH,
                include_top=True,
            )

        else:

            base = VGG16(
                weights="imagenet",
                include_top=True,
            )

    elif CNN_BACKBONE == "resnet50":

        from tensorflow.keras.applications.resnet50 import (
            ResNet50,
            preprocess_input,
        )

        base = ResNet50(
            weights="imagenet",
            include_top=True,
        )

    elif CNN_BACKBONE == "inceptionv3":

        from tensorflow.keras.applications.inception_v3 import (
            InceptionV3,
            preprocess_input,
        )

        base = InceptionV3(
            weights="imagenet",
            include_top=True,
        )

    else:

        raise ValueError(
            f"Unsupported CNN_BACKBONE: {CNN_BACKBONE}"
        )

    # VGG16 with include_top=True:
    # second-last layer = fc2 = 4096-dimensional feature vector.
    encoder = Model(
        inputs=base.inputs,
        outputs=base.layers[-2].output,
    )

    _encoder_cache["encoder"] = encoder
    _encoder_cache["preprocess"] = preprocess_input

    return (
        encoder,
        preprocess_input,
    )


# ---------------------------------------------------------------------------
# Image feature extraction
# ---------------------------------------------------------------------------
def extract_single_image_feature(
    image_path: str,
) -> np.ndarray:
    """
    Runs one image through the CNN encoder.

    Uses the same image size and preprocessing as training.
    """

    from tensorflow.keras.preprocessing.image import (
        load_img,
        img_to_array,
    )

    encoder, preprocess_input = _get_encoder()

    img = load_img(
        image_path,
        target_size=IMAGE_SIZE,
    )

    arr = img_to_array(img)

    arr = np.expand_dims(
        arr,
        axis=0,
    )

    arr = preprocess_input(arr)

    feature = encoder.predict(
        arr,
        verbose=0,
    )[0]

    return feature


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def load_tokenizer_bundle(
    path: str = TOKENIZER_PATH,
):
    """
    Loads the tokenizer bundle generated during caption preparation.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Tokenizer not found at: {path}"
        )

    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------
def generate_caption(
    model,
    feature: np.ndarray,
    tokenizer,
    max_len: int,
    max_len_cap: int = MAX_CAPTION_LEN_CAP,
) -> str:
    """
    Generates a caption using greedy decoding.

    At every step, the model selects the highest-probability next word.
    """

    from tensorflow.keras.preprocessing.sequence import (
        pad_sequences,
    )

    index_to_word = {
        idx: word
        for word, idx in tokenizer.word_index.items()
    }

    in_text = START_TOKEN

    for _ in range(max_len_cap):

        # Convert current caption text to token sequence.
        seq = tokenizer.texts_to_sequences(
            [in_text]
        )[0]

        # Pad to the sequence length used during training.
        seq = pad_sequences(
            [seq],
            maxlen=max_len,
        )

        # Add batch dimension to image feature.
        image_input = np.expand_dims(
            feature,
            axis=0,
        )

        # Predict next word.
        y_pred = model.predict(
            [
                image_input,
                seq,
            ],
            verbose=0,
        )

        # Greedy decoding.
        next_idx = int(
            np.argmax(y_pred)
        )

        next_word = index_to_word.get(
            next_idx
        )

        if next_word is None:
            break

        in_text += " " + next_word

        if next_word == END_TOKEN:
            break

    # Remove startseq/endseq tokens.
    words = [
        word
        for word in in_text.split()
        if word not in (
            START_TOKEN,
            END_TOKEN,
        )
    ]

    return " ".join(words)


# ---------------------------------------------------------------------------
# One-shot caption function
# ---------------------------------------------------------------------------
def caption_image(
    image_path: str,
    model=None,
    tokenizer_bundle=None,
) -> str:
    """
    Generates a caption for one image.
    """

    from tensorflow.keras.models import (
        load_model as keras_load_model,
    )

    # Load trained model if one was not provided.
    if model is None:

        if not os.path.isfile(
            FINAL_MODEL_PATH
        ):
            raise FileNotFoundError(
                "No trained model found at "
                f"{FINAL_MODEL_PATH}. "
                "Run src/train.py first."
            )

        model = keras_load_model(
            FINAL_MODEL_PATH
        )

    # Load tokenizer if one was not provided.
    if tokenizer_bundle is None:
        tokenizer_bundle = (
            load_tokenizer_bundle()
        )

    # Extract image feature.
    feature = (
        extract_single_image_feature(
            image_path
        )
    )

    # Generate caption.
    return generate_caption(
        model,
        feature,
        tokenizer_bundle["tokenizer"],
        tokenizer_bundle["max_len"],
    )


# ---------------------------------------------------------------------------
# Command-line usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: "
            "python src/inference.py "
            "path/to/image.jpg"
        )

        sys.exit(1)

    caption = caption_image(
        sys.argv[1]
    )

    print(
        f"Generated caption: {caption}"
    )