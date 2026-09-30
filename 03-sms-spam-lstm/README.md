# SMS Spam LSTM

## Goal
Classify SMS messages as legitimate (`ham`) or spam using a recurrent neural network.

## Dataset
**Official source:** UCI SMS Spam Collection, ID 228.

Official page: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Exact archive: https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip

## Pipeline

UCI text → train/test split → TextVectorization vocabulary → embedding layer → Bidirectional LSTM → dropout → sigmoid classifier.

## Why an LSTM?

The project demonstrates sequential representation learning before moving to transformer architectures. The same task can later be reimplemented with a transformer and compared fairly.

## Run

`pip install -r requirements.txt`

`python train.py`

The training script retrieves the dataset, learns the vocabulary only from training text, trains the model and exports a Keras model.

## Evaluation

Accuracy alone can hide spam/ham imbalance. Add precision, recall, F1 and a confusion matrix before treating the model as a useful spam filter.

## Next level

Compare vanilla LSTM, BiLSTM and DistilBERT; add threshold tuning, class weighting and adversarial/spam-template tests.