"""
model.py
--------
Phase 3: the "Merge Model" — a two-input Keras Functional API model.

    Image branch:  4096-d feature vector -> Dropout -> Dense(256, relu)
    Text branch:   padded sequence -> Embedding -> Dropout -> LSTM(256)
    Merge:         add(image_branch, text_branch)
    Output:        Dense(256, relu) -> Dense(vocab_size, softmax)

This is intentionally the classic Marc Tanti / Jason Brownlee-style merge
architecture named in the project spec (image branch -> Dense/Dropout,
text branch -> Embedding/LSTM, concatenate/add, softmax over vocab) — not
swapped out for a simpler model.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    DENSE_UNITS,
    DROPOUT_RATE,
    EMBEDDING_DIM,
    FEATURE_DIM,
    LEARNING_RATE,
    LSTM_UNITS,
)


def build_model(vocab_size: int, max_len: int, feature_dim: int = FEATURE_DIM):
    from tensorflow.keras.layers import (
        Add,
        Dense,
        Dropout,
        Embedding,
        Input,
        LSTM,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    # --- Image (encoder) branch ---------------------------------------
    image_input = Input(shape=(feature_dim,), name="image_features")
    image_branch = Dropout(DROPOUT_RATE, name="image_dropout")(image_input)
    image_branch = Dense(DENSE_UNITS, activation="relu", name="image_dense")(image_branch)

    # --- Text (partial caption) branch ---------------------------------
    text_input = Input(shape=(max_len,), name="partial_caption")
    text_branch = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True, name="caption_embedding")(text_input)
    text_branch = Dropout(DROPOUT_RATE, name="text_dropout")(text_branch)
    text_branch = LSTM(LSTM_UNITS, name="caption_lstm")(text_branch)

    # --- Merge -----------------------------------------------------------
    merged = Add(name="merge_image_text")([image_branch, text_branch])
    decoder = Dense(DENSE_UNITS, activation="relu", name="decoder_dense")(merged)
    output = Dense(vocab_size, activation="softmax", name="next_word_softmax")(decoder)

    model = Model(inputs=[image_input, text_input], outputs=output, name="VisionTalk_CNN_LSTM")
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model


def save_architecture_diagram(model, out_path: str):
    """Best-effort visualization of the merge architecture (Deliverable: Model Definition)."""
    try:
        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file=out_path, show_shapes=True, show_layer_names=True, dpi=120)
        print(f"Architecture diagram saved to {out_path}")
    except Exception as e:  # noqa: BLE001 — pydot/graphviz not installed, non-fatal
        print(f"Could not render architecture diagram ({e}). "
              f"`model.summary()` below is the fallback:")
        model.summary()


if __name__ == "__main__":
    # Quick structural smoke test with dummy sizes.
    m = build_model(vocab_size=2000, max_len=34)
    m.summary()
