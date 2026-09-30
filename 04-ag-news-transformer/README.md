# AG News Transformer Classifier

## Goal
Fine-tune a pretrained transformer for four-class news-topic classification.

## Dataset
**Hugging Face dataset:** https://huggingface.co/datasets/fancyzhx/ag_news

The dataset is retrieved through the Hugging Face Datasets library rather than committed to Git.

## Architecture

Raw article → DistilBERT tokenizer → contextual transformer encoder → sequence classification head → four-class prediction.

## Training workflow

Dataset loading → tokenization with truncation → tensor formatting → pretrained DistilBERT initialization → supervised fine-tuning → evaluation → model/tokenizer export.

## Run

`pip install -r requirements.txt`

`python train.py`

GPU is strongly recommended. Reduce batch size if running on a laptop GPU or CPU.

## What this demonstrates

Subword tokenization, attention-based contextual representations, pretrained checkpoints, fine-tuning, sequence classification and model export.

## Next level

Add explicit validation metrics, confusion matrix, learning-rate scheduling, gradient accumulation, mixed precision and inference benchmarking. Compare DistilBERT against a BiLSTM baseline.