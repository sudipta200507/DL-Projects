# Fashion-MNIST Transfer Learning

Use a pretrained MobileNetV2 backbone for Fashion-MNIST after resizing grayscale images to RGB.

Run:
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python train.py
```

The model freezes the pretrained backbone, trains a new classifier and saves `models/fashion_mobilenet.keras`.

Study feature reuse, frozen layers, fine-tuning and domain shift.