# VisionTalk — CNN + LSTM Image Captioning

Flickr8k image captioning using VGG16 transfer learning and an LSTM decoder. The image branch uses the VGG16 `fc2` 4096-D vector, while the text branch uses Embedding + LSTM. The branches are merged with Keras `Add`, followed by Dense + Softmax. Training uses teacher forcing and categorical cross-entropy.

## Setup
Python 3.11 recommended.

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

Put Flickr8k here:

```text
data\\Images\\*.jpg
data\\captions.txt
```

## Run

```bash
python src\\prepare_captions.py
python src\\extract_features.py
python src\\train.py
python src\\evaluate.py
streamlit run app.py
```

## SSL/VGG16 download issue
If automatic ImageNet weight download fails, set a local VGG16 weights path before running extraction:

```powershell
$env:VISIONTALK_VGG16_WEIGHTS="C:\\path\\to\\vgg16_weights_tf_dim_ordering_tf_kernels.h5"
```

The same setting is used by inference and Streamlit so the encoder remains consistent.

## Deliverables
- `VisionTalk_Image_Captioning.ipynb`
- BLEU-1 and BLEU-4 in `outputs/metrics.txt`
- 10-image comparison in `outputs/evaluation/10_test_images.png`
- architecture image in `outputs/model_architecture.png`
- trained model in `models/visiontalk_model.keras`

No BLEU score or model result is hard-coded; metrics are produced only by an actual evaluation run.
