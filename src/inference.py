"""
inference.py
-------------
Turns a trained VisionTalk model + a single image into a natural-language
caption. Two clearly separated pieces:

    extract_single_image_feature(path)  -> run the same CNN encoder used
                                             during training on one new image
    generate_caption(...)               -> recursive greedy-search decoding:
                                             predict one word at a time, feed
                                             it back in, stop at endseq or
                                             MAX_CAPTION_LEN_CAP.

This module is imported by both evaluate.py (batch evaluation) and the
Streamlit app (single-image demo) so decoding logic only lives in one place.
"""

import os
import pickle
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    CNN_BACKBONE,
    END_TOKEN,
    FINAL_MODEL_PATH,
    IMAGE_SIZE,
    MAX_CAPTION_LEN_CAP,
    START_TOKEN,
    TOKENIZER_PATH,
)

_encoder_cache = {}


def _get_encoder():
    """Lazily builds and caches the same feature-extractor CNN used in training."""
    if "encoder" in _encoder_cache:
        return _encoder_cache["encoder"], _encoder_cache["preprocess"]

    from tensorflow.keras.models import Model

    if CNN_BACKBONE == "vgg16":
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
        base = VGG16(weights="imagenet")
    elif CNN_BACKBONE == "resnet50":
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        base = ResNet50(weights="imagenet", include_top=True)
    elif CNN_BACKBONE == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        base = InceptionV3(weights="imagenet", include_top=True)
    else:
        raise ValueError(f"Unsupported CNN_BACKBONE: {CNN_BACKBONE}")

    encoder = Model(inputs=base.inputs, outputs=base.layers[-2].output)
    _encoder_cache["encoder"] = encoder
    _encoder_cache["preprocess"] = preprocess_input
    return encoder, preprocess_input


def extract_single_image_feature(image_path: str) -> np.ndarray:
    """Runs one image through the CNN encoder. Same preprocessing as training."""
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    encoder, preprocess_input = _get_encoder()
    img = load_img(image_path, target_size=IMAGE_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return encoder.predict(arr, verbose=0)[0]


def load_tokenizer_bundle(path: str = TOKENIZER_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)  # {"tokenizer", "vocab_size", "max_len"}


def generate_caption(model, feature: np.ndarray, tokenizer, max_len: int,
                      max_len_cap: int = MAX_CAPTION_LEN_CAP) -> str:
    """
    Greedy search: at each step, feed the image feature + the caption-so-far,
    take the single highest-probability next word, append it, and repeat
    until endseq is produced or we hit the length safety cap.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    index_to_word = {idx: w for w, idx in tokenizer.word_index.items()}

    in_text = START_TOKEN
    for _ in range(max_len_cap):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len)
        y_pred = model.predict([np.expand_dims(feature, axis=0), seq], verbose=0)
        next_idx = int(np.argmax(y_pred))
        next_word = index_to_word.get(next_idx)

        if next_word is None:
            break
        in_text += " " + next_word
        if next_word == END_TOKEN:
            break

    # Strip control tokens for the human-readable caption.
    words = [w for w in in_text.split() if w not in (START_TOKEN, END_TOKEN)]
    return " ".join(words)


def caption_image(image_path: str, model=None, tokenizer_bundle=None) -> str:
    """Convenience one-shot entry point used by the Streamlit app."""
    from tensorflow.keras.models import load_model as keras_load_model

    if model is None:
        if not os.path.isfile(FINAL_MODEL_PATH):
            raise FileNotFoundError(f"No trained model found at {FINAL_MODEL_PATH}. Run src/train.py first.")
        model = keras_load_model(FINAL_MODEL_PATH)
    if tokenizer_bundle is None:
        tokenizer_bundle = load_tokenizer_bundle()

    feature = extract_single_image_feature(image_path)
    return generate_caption(
        model, feature, tokenizer_bundle["tokenizer"], tokenizer_bundle["max_len"]
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/inference.py path/to/image.jpg")
        sys.exit(1)
    caption = caption_image(sys.argv[1])
    print(f"Generated caption: {caption}")
