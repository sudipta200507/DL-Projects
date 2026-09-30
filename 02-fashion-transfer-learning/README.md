# Fashion-MNIST Transfer Learning

## Goal
Use a pretrained visual representation instead of training an image model entirely from random initialization.

## Dataset
**Official TensorFlow/Keras source:** https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist

Fashion-MNIST provides grayscale 28×28 clothing images across 10 classes.

## Architecture

28×28 grayscale → resize to 96×96 → grayscale-to-RGB → pretrained MobileNetV2 backbone → global average pooling → dropout → 10-class classifier.

The backbone is initially frozen so the experiment isolates transfer learning from full network training.

## Run

`pip install -r requirements.txt`

`python train.py`

ImageNet weights are downloaded automatically by Keras. The Fashion-MNIST data is also downloaded automatically. Generated weights are ignored by Git.

## What this demonstrates

Transfer learning, pretrained representations, frozen parameters, domain mismatch, classifier heads and the difference between feature extraction and fine-tuning.

## Next level

Unfreeze the last MobileNet blocks, use a lower learning rate, add augmentation and compare frozen-backbone versus fine-tuned performance.