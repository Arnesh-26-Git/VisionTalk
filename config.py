from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; IMAGES=DATA/'Images'; CAPTIONS=DATA/'captions.txt'; MODELS=ROOT/'models'; OUTPUTS=ROOT/'outputs'
BACKBONE='vgg16'; IMAGE_SIZE=(224,224); FEATURE_DIM=4096
VGG16_WEIGHTS_PATH=os.environ.get('VISIONTALK_VGG16_WEIGHTS','')
MAX_VOCAB_SIZE=10000; EMBEDDING_DIM=256; LSTM_UNITS=256; DENSE_UNITS=256; DROPOUT=0.5
BATCH_SIZE=32; EPOCHS=20; TRAIN_SPLIT=.85; SEED=42; BEAM_WIDTH=3
for d in (MODELS,OUTPUTS,OUTPUTS/'evaluation'): d.mkdir(parents=True,exist_ok=True)
