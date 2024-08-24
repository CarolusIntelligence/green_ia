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
import statsmodels as statsmodels import os


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
    for batch in train_batches:
        # passerbatch au modèle deep learning
        # model.train_on_batch(batch)
        pass

if __name__ == "__main__":
    file_id = sys.argv[1]
    data_path = sys.argv[2]
    main(file_id, data_path)