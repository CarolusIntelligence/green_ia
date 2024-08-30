import numpy as np
import pandas as pd
import jsonlines
import os
import warnings
from datetime import datetime 
import json
import re
import random
import sys
import math
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from io import StringIO 

pd.set_option('display.max_rows', 100)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


def read_jsonl_in_chunks(file_path, chunk_size):
    chunks = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            chunks.append(pd.read_json(StringIO(line), lines=True))
            if (i + 1) % chunk_size == 0:
                yield pd.concat(chunks, ignore_index=True)
                chunks = []
    if chunks:
        yield pd.concat(chunks, ignore_index=True)

def data_to_tensor(chunk):
    y = torch.tensor(chunk['ecoscore_score'].values, dtype=torch.float32)

    # convertir colonnes textuelles en string
    chunk['packaging'] = chunk['packaging'].astype(str)
    chunk['name'] = chunk['name'].astype(str)
    chunk['ingredients'] = chunk['ingredients'].astype(str)
    chunk['categories'] = chunk['categories'].astype(str)

    # encoder variables textuelles
    vectorizer = CountVectorizer()
    encoded_packaging = vectorizer.fit_transform(chunk['packaging']).toarray()
    encoded_name = vectorizer.fit_transform(chunk['name']).toarray()
    encoded_ingredients = vectorizer.fit_transform(chunk['ingredients']).toarray()
    encoded_categories = vectorizer.fit_transform(chunk['categories']).toarray()

    # concatener colonnes encodees avec les colonnes numeriques
    x = pd.concat([
        chunk[['groups', 'countries', 'labels_note']].reset_index(drop=True),
        pd.DataFrame(encoded_packaging, columns=[f'packaging_{i}' for i in range(encoded_packaging.shape[1])]),
        pd.DataFrame(encoded_name, columns=[f'name_{i}' for i in range(encoded_name.shape[1])]),
        pd.DataFrame(encoded_ingredients, columns=[f'ingredients_{i}' for i in range(encoded_ingredients.shape[1])]),
        pd.DataFrame(encoded_categories, columns=[f'categories_{i}' for i in range(encoded_categories.shape[1])])
    ], axis=1)
    # convertir dataframe en tenseur torch
    x = torch.tensor(x.values, dtype=torch.float32)

    return x, y



###############################################################################
# MAIN ########################################################################
###############################################################################
def main(chunk_size, file_id, data_path):
    chunk_size = int(chunk_size)
    train = data_path + file_id + "_train" + ".jsonl"
    test = data_path + file_id + "_test" + ".jsonl"
    valid = data_path + file_id + "_valid" + ".jsonl"

    print("creating tensor")
    for chunk in read_jsonl_in_chunks(train, chunk_size):
        x, y = data_to_tensor(chunk)
        print('tensor x :', x)
        print('tensor y :', y)


if __name__ == "__main__":
    chunk_size = sys.argv[1]
    file_id = sys.argv[2]
    data_path = sys.argv[3]
    main(chunk_size, file_id, data_path)