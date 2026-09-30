# SMS Spam LSTM

Train a bidirectional LSTM to classify SMS messages as ham or spam.

Dataset: UCI SMS Spam Collection (ID 228).

Run `pip install -r requirements.txt` then `python train.py`. The script downloads the dataset, builds a TextVectorization vocabulary, trains the LSTM and saves `models/sms_spam_lstm.keras`.

Study tokenization, embeddings, sequence modeling, bidirectional recurrence, class imbalance and precision/recall.