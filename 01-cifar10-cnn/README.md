# CIFAR-10 CNN Image Classifier

Train a convolutional neural network to classify 10 object categories.

Dataset: CIFAR-10 through TensorFlow Datasets.

```bash
git clone https://github.com/sudipta200507/DL-Projects.git
cd DL-Projects/01-cifar10-cnn
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python train.py
```

The script downloads the public dataset, normalizes images, trains a CNN, evaluates the test set and saves `models/cifar10.keras`.

Study convolution, kernels, pooling, receptive fields, augmentation and overfitting.