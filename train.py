import sys
from pathlib import Path
import pickle
import math
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Embedding,
    LSTM,
    Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

FEATURES_FILE = MODELS_DIR / "features.pkl"
CAPTIONS_FILE = MODELS_DIR / "clean_captions.pkl"
SPLITS_FILE = MODELS_DIR / "splits.pkl"
TOKENIZER_FILE = MODELS_DIR / "tokenizer.pkl"

MODEL_FILE = MODELS_DIR / "visiontalk_model.keras"

IMAGE_FEATURE_SIZE = 4096
EMBEDDING_DIM = 256
LSTM_UNITS = 256
DROPOUT = 0.3

BATCH_SIZE = 32
EPOCHS = 20

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data():

    print("Loading VisionTalk data...")

    with open(FEATURES_FILE, "rb") as f:
        features = pickle.load(f)

    with open(CAPTIONS_FILE, "rb") as f:
        captions = pickle.load(f)

    with open(SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)

    with open(TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)

    word_index = tokenizer["word_index"]

    vocab_size = len(word_index)

    max_length = max(
        len(caption.split())
        for image_captions in captions.values()
        for caption in image_captions
    )

    print(f"Features: {len(features)}")
    print(f"Vocabulary: {vocab_size}")
    print(f"Maximum length: {max_length}")

    return (
        features,
        captions,
        splits,
        word_index,
        vocab_size,
        max_length
    )


def create_samples(
    features,
    captions,
    image_ids,
    word_index,
    max_length
):

    samples = []

    unk = word_index["<unk>"]

    for image_id in image_ids:

        if image_id not in features:
            continue

        for caption in captions[image_id]:

            words = caption.split()

            sequence = [
                word_index.get(word, unk)
                for word in words
            ]

            for i in range(1, len(sequence)):

                input_sequence = sequence[:i]
                target_word = sequence[i]

                if len(input_sequence) > max_length:
                    input_sequence = input_sequence[
                        -max_length:
                    ]

                padded = np.zeros(
                    max_length,
                    dtype=np.int32
                )

                padded[
                    :len(input_sequence)
                ] = input_sequence

                samples.append(
                    (
                        image_id,
                        padded,
                        target_word
                    )
                )

    return samples


class CaptionSequence(tf.keras.utils.Sequence):

    def __init__(
        self,
        samples,
        features,
        batch_size
    ):

        super().__init__()

        self.samples = samples
        self.features = features
        self.batch_size = batch_size

    def __len__(self):

        return math.ceil(
            len(self.samples) /
            self.batch_size
        )

    def __getitem__(self, index):

        batch_samples = self.samples[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        image_batch = np.empty(
            (
                len(batch_samples),
                IMAGE_FEATURE_SIZE
            ),
            dtype=np.float32
        )

        text_batch = np.empty(
            (
                len(batch_samples),
                batch_samples[0][1].shape[0]
            ),
            dtype=np.int32
        )

        target_batch = np.empty(
            len(batch_samples),
            dtype=np.int32
        )

        for i, (
            image_id,
            text,
            target
        ) in enumerate(batch_samples):

            image_batch[i] = self.features[
                image_id
            ]

            text_batch[i] = text
            target_batch[i] = target

        return (
            {
                "image_input": image_batch,
                "text_input": text_batch
            },
            target_batch
        )


def build_model(
    vocab_size,
    max_length
):

    image_input = Input(
        shape=(IMAGE_FEATURE_SIZE,),
        name="image_input"
    )

    image_branch = Dense(
        LSTM_UNITS,
        activation="relu"
    )(image_input)

    image_branch = Dropout(
        DROPOUT
    )(image_branch)

    text_input = Input(
        shape=(max_length,),
        name="text_input"
    )

    text_branch = Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    )(text_input)

    text_branch = LSTM(
        LSTM_UNITS
    )(text_branch)

    merged = Add()([
        image_branch,
        text_branch
    ])

    merged = Dense(
        LSTM_UNITS,
        activation="relu"
    )(merged)

    merged = Dropout(
        DROPOUT
    )(merged)

    output = Dense(
        vocab_size,
        activation="softmax",
        name="word_output"
    )(merged)

    model = Model(
        inputs=[
            image_input,
            text_input
        ],
        outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            "features.pkl not found. "
            "Run extract_features.py first."
        )

    (
        features,
        captions,
        splits,
        word_index,
        vocab_size,
        max_length
    ) = load_data()

    print()
    print("Creating training samples...")

    train_samples = create_samples(
        features,
        captions,
        splits["train"],
        word_index,
        max_length
    )

    print(
        f"Training samples: {len(train_samples)}"
    )

    print("Creating validation samples...")

    val_samples = create_samples(
        features,
        captions,
        splits["val"],
        word_index,
        max_length
    )

    print(
        f"Validation samples: {len(val_samples)}"
    )

    print()
    print("Building model...")

    model = build_model(
        vocab_size,
        max_length
    )

    model.summary()

    train_generator = CaptionSequence(
        train_samples,
        features,
        BATCH_SIZE
    )

    val_generator = CaptionSequence(
        val_samples,
        features,
        BATCH_SIZE
    )

    checkpoint = ModelCheckpoint(
        MODEL_FILE,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    print()
    print("=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )

    model.save(MODEL_FILE)

    print()
    print("=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print(
        f"Model saved to:\n{MODEL_FILE}"
    )


if __name__ == "__main__":
    main()