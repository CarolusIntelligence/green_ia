import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sea
import sklearn
import scipy as sc
import nltk as nltk
import statsmodels as statsmodels
import os
import warnings
import csv
from datetime import datetime 
import json

pd.set_option('display.max_rows', 50)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import dask
import random


print(dask.__version__)



# configuration
chunk_size = 5000
file_id = '00'
project_path = "/home/carolus/Documents/school/green_ia/" 
jsonl = project_path + "data/" + file_id + "_openfoodfacts" + ".jsonl"
jsonl_filtered = project_path + 'data/' + file_id + '_openfoodfacts_filtered.jsonl'
jsonl_sample = project_path + 'data/' + file_id + '_openfoodfacts_sample.jsonl'
col_to_keep = ['allergens_from_ingredients',
               'pnns_groups_1',
               'ecoscore_data',
               'ingredients_tags',
               'packaging',
               'product_name',
               'food_groups_tags',
               'ecoscore_tags',
               'categories_tags',
               'ecoscore_score',
               'labels_tags',
               'countries']



def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"file deleted: {file_path}")
    else:
        print(f"ERROR, does not exists: {file_path}")


# génération jsonl filtré
def jsonl_filtered_creator():
    with open(jsonl, 
              'r', 
              encoding='utf-8') as infile, open(jsonl_filtered, 'w', 
                                                encoding='utf-8') as outfile:
        buffer = []
        
        for i, line in enumerate(infile):
            record = json.loads(line.strip())        
            filtered_record = {key: record.get(key) for key in col_to_keep}        
            buffer.append(json.dumps(filtered_record) + '\n')
            
            if len(buffer) >= chunk_size:
                outfile.writelines(buffer)
                buffer = []
        
        if buffer:
            outfile.writelines(buffer)


# création d'un échantillion (mini jsonl) pour inspecter la qualité des données
#  (sélection aléatoire de 1000 lignes )
def jsonl_sample_creator():
    sampled_rows = []

    for chunk in pd.read_json(jsonl_filtered, lines=True, chunksize=chunk_size):
        sampled_chunk = chunk.sample(n=min(1000, len(chunk)), 
                                     random_state=random.randint(1, 10000))
        sampled_rows.append(sampled_chunk)

    sampled_df = pd.concat(sampled_rows)

    if len(sampled_df) > 1000:
        sampled_df = sampled_df.sample(n=1000, random_state=42)

    sampled_df.to_json(jsonl_sample, orient='records', lines=True)
    print(f"jsonl sample created: {jsonl_sample}")



# main algo
jsonl_filtered_creator()
delete_file(jsonl)
jsonl_sample_creator()
