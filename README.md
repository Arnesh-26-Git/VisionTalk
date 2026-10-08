# VisionTalk — VGG16 + LSTM Image Captioning

VisionTalk is an image-captioning application built using the Flickr8k dataset. It uses VGG16 transfer learning for visual feature extraction and an LSTM decoder to generate captions word by word.

## Architecture

1. **Image encoder:** VGG16 `fc2` 4096-dimensional feature vector.
2. **Text decoder:** Embedding + LSTM.
3. **Fusion:** Keras Functional API element-wise `Add`.
4. **Output:** Dense + Softmax next-word prediction.
5. **Inference:** greedy next-word decoding.

## Project structure

```text
VisionTalk/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── Images/
│   └── captions.txt
├── models/
│   ├── features.pkl
│   ├── tokenizer.pkl
│   └── visiontalk_model.keras
├── outputs/
│   ├── bleu_scores.json
│   ├── metrics.txt
│   ├── predictions_test.csv
│   ├── sample_predictions.png
│   └── evaluation/
│       └── 10_test_images.png
├── src/
├── VisionTalk_Image_Captioning.ipynb
├── config.py
├── requirements.txt
└── vgg16_weights.h5
```

## Setup

Python 3.11 is recommended.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The Flickr8k dataset is expected at:

```text
data\Images\*.jpg
data\captions.txt
```

If the VGG16 automatic weight download fails, a local VGG16 weight file can be supplied through the `VISIONTALK_VGG16_WEIGHTS` environment variable.

## Reproduce the pipeline

Run these commands from the project root:

```powershell
python src\prepare_captions.py
python src\extract_features.py
python src\train.py
python src\evaluate.py
```

Do not run the training command again when only the finished demo is required; the completed project already contains the trained model and evaluation artifacts.

## Streamlit demo

The actual application entrypoint is:

```powershell
streamlit run app\streamlit_app.py
```

The app accepts:

- JPG
- JPEG
- PNG

It displays the uploaded image and generates a caption using the trained VisionTalk model.

## Final evaluation

The completed evaluation scored **809 test images**.

```text
BLEU-1: 0.5008
BLEU-4: 0.1413
Test images scored: 809
```

Generated evaluation artifacts:

```text
outputs\bleu_scores.json
outputs\predictions_test.csv
outputs\metrics.txt
outputs\sample_predictions.png
outputs\evaluation\10_test_images.png
```

`10_test_images.png` contains a qualitative comparison of generated captions against reference captions for 10 test images.

## Final deliverables

- `VisionTalk_Image_Captioning.ipynb`
- `outputs/metrics.txt`
- `outputs/evaluation/10_test_images.png`
- `models/visiontalk_model.keras`
- Streamlit demo through `app/streamlit_app.py`

The reported model and evaluation results are generated from the Flickr8k dataset; they are not fabricated or hard-coded.

## Limitations

The model is trained on Flickr8k, so it performs best on images similar to that dataset. It is not an OCR system and is not designed to reliably read text from screenshots or documents. Caption wording can also be imperfect for complex scenes or actions.

## Final status

**VisionTalk is complete and ready for demonstration/submission.**
