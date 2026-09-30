# AG News Transformer Classifier

Fine-tune a pretrained transformer for four-way news-topic classification.

Dataset: AG News via Hugging Face Datasets.

Run:
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python train.py
```

The script tokenizes the dataset, fine-tunes a compact DistilBERT classifier and saves the model/tokenizer under `models/`.

Study attention, tokenization, contextual embeddings, fine-tuning and evaluation.