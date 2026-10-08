"""
train.py
--------
Phase 4: trains the VisionTalk CNN+LSTM merge model using the memory-
efficient data generator (teacher forcing), with model checkpointing after
every epoch and a final saved model + training history.

Assumes you have already run, in order:
    python src/download_dataset.py
    python src/prepare_captions.py
    python src/extract_features.py

Usage:
    python src/train.py
    EPOCHS=30 BATCH_SIZE=32 python src/train.py     # override hyperparameters
"""

import json
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    BATCH_SIZE,
    CLEAN_CAPTIONS_PATH,
    CONFIG_SNAPSHOT_PATH,
    DENSE_UNITS,
    DROPOUT_RATE,
    EMBEDDING_DIM,
    EPOCHS,
    FEATURES_PATH,
    FINAL_MODEL_PATH,
    HISTORY_PATH,
    LEARNING_RATE,
    LSTM_UNITS,
    MODEL_CHECKPOINT_DIR,
    OUTPUTS_DIR,
    SPLITS_PATH,
    TOKENIZER_PATH,
)
from data_generator import CaptionDataGenerator  # noqa: E402
from model import build_model, save_architecture_diagram  # noqa: E402


def _load_pickled_artifacts():
    missing = [p for p in (CLEAN_CAPTIONS_PATH, TOKENIZER_PATH, SPLITS_PATH, FEATURES_PATH) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Missing preprocessing artifacts: " + ", ".join(missing) +
            "\nRun, in order: src/prepare_captions.py then src/extract_features.py"
        )

    with open(CLEAN_CAPTIONS_PATH, "rb") as f:
        clean_captions = pickle.load(f)
    with open(TOKENIZER_PATH, "rb") as f:
        tok_bundle = pickle.load(f)
    with open(SPLITS_PATH, "rb") as f:
        splits = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)

    return clean_captions, tok_bundle, splits, features


def run():
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    clean_captions, tok_bundle, splits, features = _load_pickled_artifacts()
    tokenizer = tok_bundle["tokenizer"]
    vocab_size = tok_bundle["vocab_size"]
    max_len = tok_bundle["max_len"]

    print(f"Vocab size: {vocab_size} | Max caption length: {max_len}")
    print(f"Train images: {len(splits['train'])} | Val images: {len(splits['val'])} | Test images: {len(splits['test'])}")

    train_gen = CaptionDataGenerator(
        splits["train"], clean_captions, features, tokenizer, max_len, vocab_size,
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_gen = CaptionDataGenerator(
        splits["val"], clean_captions, features, tokenizer, max_len, vocab_size,
        batch_size=BATCH_SIZE, shuffle=False,
    )

    feature_dim = next(iter(features.values())).shape[0]
    model = build_model(vocab_size=vocab_size, max_len=max_len, feature_dim=feature_dim)
    model.summary()
    save_architecture_diagram(model, os.path.join(OUTPUTS_DIR, "model_architecture.png"))

    checkpoint_path = os.path.join(MODEL_CHECKPOINT_DIR, "epoch_{epoch:02d}_valloss_{val_loss:.3f}.keras")
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=False, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(FINAL_MODEL_PATH)
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)

    run_config = {
        "epochs_requested": EPOCHS,
        "epochs_ran": len(history.history.get("loss", [])),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
        "dense_units": DENSE_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "vocab_size": vocab_size,
        "max_len": max_len,
        "final_train_loss": history.history["loss"][-1] if history.history.get("loss") else None,
        "final_val_loss": history.history["val_loss"][-1] if history.history.get("val_loss") else None,
    }
    with open(CONFIG_SNAPSHOT_PATH, "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nSaved final model to {FINAL_MODEL_PATH}")
    print(f"Saved training history to {HISTORY_PATH}")
    print(f"Saved run config/summary to {CONFIG_SNAPSHOT_PATH}")
    return model, history


if __name__ == "__main__":
    run()
