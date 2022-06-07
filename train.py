import logging
import numpy as np
from datasets import load_dataset
from keras.preprocessing.sequence import pad_sequences
from transformers import RobertaTokenizer

import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Embedding, Flatten, GlobalAveragePooling1D, Dense, Bidirectional, LSTM

import mlflow
import mlflow.tensorflow

mlflow.set_tracking_uri('https://community.mlflow.deploif.ai')
mlflow.set_experiment("Deploifai/hugging-face-sentiment-analysis/hugging-face-training")

logger = logging.getLogger(__name__)

def process_dataset(dataset):
  tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
  sentences = [d.as_py() for d in dataset.data['text']]
  labels = np.array([l.as_py() for l in dataset.data['label']])
  sequences = [tokenizer(s)['input_ids'] for s in sentences]
  padded = pad_sequences(sequences=sequences, padding='post', maxlen=35)
  return padded, labels

mlflow.tensorflow.autolog(log_models=False)

train_dataset = load_dataset("tweet_eval", "emoji", split='train')
test_dataset = load_dataset("tweet_eval", "emoji", split='test')

logger.info("Fetched database")

training_padded, training_labels = process_dataset(train_dataset)
test_padded, test_labels = process_dataset(test_dataset)

model = Sequential([
  Embedding(50265, 64, input_length=35), # Roberta tokeniser vocabulary size is 50,265
  Bidirectional(LSTM(64, return_sequences=True)),
  Bidirectional(LSTM(32)),
  Dense(512, activation='relu'),
  Dense(20, activation='softmax') # The emoji dataset has 20 classes
])

model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()

model.fit(training_padded, training_labels, epochs=100, validation_data=(test_padded, test_labels))
model.save("model")
