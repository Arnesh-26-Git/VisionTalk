"""
config.py
---------
Single source of truth for every path and hyperparameter used across the
VisionTalk project (data prep, feature extraction, training, inference,
evaluation, and the Streamlit app).

Nothing in this project hardcodes a personal/local path — every script
imports from here. Override any value with an environment variable of the
same name (e.g. `EPOCHS=30 python src/train.py`) without touching code.
"""

import os

# ---------------------------------------------------------------------------
# Paths (all relative to the project root; safe on Colab, Windows, macOS, Linux)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get("VISIONTALK_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
IMAGES_DIR = os.path.join(DATA_DIR, "Images")                     # Flickr8k images
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.txt")            # Flickr8k captions.txt

MODELS_DIR = os.environ.get("VISIONTALK_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
OUTPUTS_DIR = os.environ.get("VISIONTALK_OUTPUTS_DIR", os.path.join(PROJECT_ROOT, "outputs"))

FEATURES_PATH = os.path.join(MODELS_DIR, "features.pkl")          # CNN feature vectors
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")
CLEAN_CAPTIONS_PATH = os.path.join(MODELS_DIR, "clean_captions.pkl")
SPLITS_PATH = os.path.join(MODELS_DIR, "splits.pkl")
MODEL_CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "visiontalk_model.keras")
HISTORY_PATH = os.path.join(MODELS_DIR, "history.pkl")
CONFIG_SNAPSHOT_PATH = os.path.join(MODELS_DIR, "run_config.json")

for _d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, MODEL_CHECKPOINT_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# CNN encoder
# ---------------------------------------------------------------------------
CNN_BACKBONE = os.environ.get("CNN_BACKBONE", "vgg16")   # vgg16 | resnet50 | inceptionv3
IMAGE_SIZE = {
    "vgg16": (224, 224),
    "resnet50": (224, 224),
    "inceptionv3": (299, 299),
}[CNN_BACKBONE]
FEATURE_DIM = {
    "vgg16": 4096,
    "resnet50": 2048,
    "inceptionv3": 2048,
}[CNN_BACKBONE]

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
START_TOKEN = "startseq"
END_TOKEN = "endseq"
OOV_TOKEN = "<unk>"
MIN_WORD_FREQ = int(os.environ.get("MIN_WORD_FREQ", 2))   # drop rare words from vocab

# ---------------------------------------------------------------------------
# Model / training hyperparameters (all overridable via env vars)
# ---------------------------------------------------------------------------
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 256))
LSTM_UNITS = int(os.environ.get("LSTM_UNITS", 256))
DENSE_UNITS = int(os.environ.get("DENSE_UNITS", 256))
DROPOUT_RATE = float(os.environ.get("DROPOUT_RATE", 0.5))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
EPOCHS = int(os.environ.get("EPOCHS", 20))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-3))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", 0.1))
TEST_SPLIT = float(os.environ.get("TEST_SPLIT", 0.1))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", 42))

# Inference
BEAM_WIDTH = int(os.environ.get("BEAM_WIDTH", 1))   # 1 == greedy search
MAX_CAPTION_LEN_CAP = int(os.environ.get("MAX_CAPTION_LEN_CAP", 35))  # safety cap

# Evaluation
N_EVAL_VISUALS = int(os.environ.get("N_EVAL_VISUALS", 10))
