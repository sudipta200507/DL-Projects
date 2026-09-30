# CIFAR-10 CNN Image Classifier

## Goal
Build a convolutional neural network for ten-class image classification.

## Dataset
**Official source:** TensorFlow Datasets CIFAR-10: https://www.tensorflow.org/datasets/catalog/cifar10
**Original dataset:** https://www.cs.toronto.edu/~kriz/cifar.html

CIFAR-10 contains 60,000 32×32 RGB images: 50,000 train and 10,000 test, across 10 classes. citeturn0search2

## Architecture
Input → Conv2D → MaxPooling → Conv2D → MaxPooling → Conv2D → Flatten → Dense → Dropout → Softmax.

## Engineering workflow

Dataset loader → normalization → shuffled batched pipeline → CNN training → validation/test evaluation → Keras model export.

## Run

`pip install -r requirements.txt`

`python train.py`

The dataset is downloaded automatically by TensorFlow Datasets. No dataset is committed to Git.

## What to inspect

Study tensor shapes, convolution kernels, receptive fields, pooling, parameter counts, training/validation curves and overfitting.

## Next level

Add augmentation, BatchNorm, learning-rate scheduling, EarlyStopping, confusion matrices, per-class metrics and TensorBoard experiment tracking.