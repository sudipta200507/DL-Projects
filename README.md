# Deep Learning Projects

A reproducible deep-learning laboratory covering computer vision, NLP, transformers and time-series learning.

## Project map

| Project | Domain | Architecture | Dataset/source |
|---|---|---|---|
| 01 | Vision | CNN | CIFAR-10 / TensorFlow Datasets |
| 02 | Vision | MobileNetV2 transfer learning | Fashion-MNIST |
| 03 | NLP | BiLSTM | UCI SMS Spam Collection |
| 04 | NLP | DistilBERT | AG News / Hugging Face Datasets |
| 05 | Time series | LSTM | UCI Bike Sharing |

## Exact dataset sources

- CIFAR-10 — TensorFlow Datasets: https://www.tensorflow.org/datasets/catalog/cifar10 — original dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Fashion-MNIST — TensorFlow/Keras: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist
- SMS Spam Collection — UCI ID 228: https://archive.ics.uci.edu/dataset/228/sms+spam+collection — direct archive: https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
- AG News — Hugging Face Datasets: https://huggingface.co/datasets/fancyzhx/ag_news
- Bike Sharing — UCI ID 275: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset — direct archive: https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip

CIFAR-10 contains 60,000 32×32 colour images across 10 classes, with 50,000 training and 10,000 test examples. citeturn0search2

## Workflow

**dataset → input pipeline → preprocessing → architecture → training → validation → evaluation → checkpoint/export → inference**

Large datasets, checkpoints and generated model files are not committed to the repository.

## Hardware

The vision and LSTM projects can run on CPU, although training is faster on a GPU. The transformer project benefits substantially from a GPU and may need smaller batch sizes on a laptop.

## What to study

For every project, inspect tensor shapes, loss functions, optimizers, validation behaviour, overfitting, checkpointing and inference. After the baseline works, add augmentation, learning-rate scheduling, early stopping, mixed precision and experiment tracking.

## Reproducibility

Every project has its own requirements file and README. Run the project locally instead of relying on committed screenshots or pre-generated metrics.
