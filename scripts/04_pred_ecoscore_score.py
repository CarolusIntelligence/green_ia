import pandas as pd
import numpy as np
import tensorflow as tf
import sys 
from io import StringIO
import keras
import math
import matplotlib.pyplot as plt
import seaborn as sea
import sklearn
import scipy as sc
import nltk as nltk
import statsmodels as statsmodels 
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler


def setup_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("increasing gpu perf ok")
        except RuntimeError as e:
            print(f"ERROR, {e}")
setup_gpu()

def read_jsonl_in_batches(file_path, batch_size):
    batch = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            batch.append(pd.read_json(line, lines=True))
            if (i + 1) % batch_size == 0:
                yield pd.concat(batch, ignore_index=True)
                batch = []
        if batch:
            yield pd.concat(batch, ignore_index=True)

def load_data_in_batches(train, test, valid, batch_size=1000):
    train_batches = read_jsonl_in_batches(train, batch_size)
    valid_batches = read_jsonl_in_batches(valid, batch_size)
    test_batches = read_jsonl_in_batches(test, batch_size)
    return train_batches, valid_batches, test_batches

def training(train_batches, test_batches):
    label_encoders = {}
    is_first_batch = True
    scaler = StandardScaler()
    for batch in train_batches:
        y = batch['ecoscore_score'].values
        x = batch[['groups', 'packaging', 'name', 'countries', 'ingredients', 'categories', 'labels-note']]
        # encodage colonnes catégorielles
        if is_first_batch:
            for column in x.select_dtypes(include=['object']).columns:
                label_encoders[column] = LabelEncoder()
                x[column] = label_encoders[column].fit_transform(x[column])
            x_scaled = scaler.fit_transform(x)
            is_first_batch = False
        else:
            # appliquer encodage avec encodeurs déjà appris
            for column in x.select_dtypes(include=['object']).columns:
                x[column] = label_encoders[column].transform(x[column])
            x_scaled = scaler.transform(x)

        x_train = tf.convert_to_tensor(x_scaled, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y, dtype=tf.float32)
        
        # Passer le lot traité (x_train, y_train) au modèle de deep learning
        # model.train_on_batch(x_train, y_train)
        pass

    # prétraitement lots de test
    for batch in test_batches:
        batch.fillna('Unknown', inplace=True)
        y = batch['ecoscore_score'].values
        x = batch[['groups', 'packaging', 'name', 'countries', 'ingredients', 'categories', 'labels-note']]
        for column in x.select_dtypes(include=['object']).columns:
            x[column] = label_encoders[column].transform(x[column])
        x_scaled = scaler.transform(x)
        x_test = tf.convert_to_tensor(x_scaled, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y, dtype=tf.float32)
        # Utilisez x_test et y_test pour évaluer le modèle
        # model.evaluate(x_test, y_test)
        pass  



###############################################################################
# MAIN ########################################################################
###############################################################################
def main(file_id, data_path):
    train = data_path + file_id + "_train" + ".jsonl"
    test = data_path + file_id + "_test" + ".jsonl"
    valid = data_path + file_id + "_valid" + ".jsonl"
    save_model = data_path + '../models/' + file_id + "_model" + ".ci"
    print("setup gpu")
    setup_gpu()
    train_batches, valid_batches, test_batches = load_data_in_batches(train, test, valid)
    training(train_batches, test_batches)

if __name__ == "__main__":
    file_id = sys.argv[1]
    data_path = sys.argv[2]
    main(file_id, data_path)