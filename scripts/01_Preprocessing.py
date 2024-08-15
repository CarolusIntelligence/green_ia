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
import random




# configuration
chunk_size = 5000
file_id = '02'
project_path = "/home/carolus/Documents/school/green_ia/" 
jsonl_00 = project_path + "data/" + file_id + "_openfoodfacts_00" + ".jsonl" # fichier sans aucune étape de prétraitement (dézipé) 
jsonl_01 = project_path + 'data/' + file_id + '_openfoodfacts_01.jsonl' # fichier avec première étape de prétraitement (uniquement colonnes intéressantes)
jsonl_02 = project_path + 'data/' + file_id + '_openfoodfacts_02.jsonl' # fichier avec deuxième étape de prétraitement (traitement intégral)
jsonl_sample = project_path + 'data/' + file_id + '_openfoodfacts_sample.jsonl'
col_to_keep = ['pnns_groups_1',
               'ingredients_tags',
               'packaging',
               'product_name',
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
def jsonl_filtered_creator(origin_file):
    with open(origin_file, 
              'r', 
              encoding='utf-8') as infile, open(jsonl_01, 'w', 
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
def jsonl_sample_creator(file_to_sample, jsonl_sample):
    print(f"{file_to_sample}, {jsonl_sample}")
    
    sampled_rows = []

    for chunk in pd.read_json(file_to_sample, lines=True, chunksize=chunk_size):
        sampled_chunk = chunk.sample(n=min(1000, len(chunk)), 
                                     random_state=random.randint(1, 10000))
        sampled_rows.append(sampled_chunk)

    sampled_df = pd.concat(sampled_rows)

    if len(sampled_df) > 1000:
        sampled_df = sampled_df.sample(n=1000, random_state=42)

    sampled_df.to_json(jsonl_sample, orient='records', lines=True)
    print(f"jsonl sample created: {jsonl_sample}")


def main_processing(jsonl_01, jsonl_02):
    print("a developper")



# main algo
jsonl_filtered_creator(jsonl_00)
#delete_file(jsonl_00)
#main_processing(jsonl_01, jsonl_02)
#delete_file(jsonl_01)
jsonl_sample_creator(jsonl_01, jsonl_sample) # puis utiliser 02 car prétraitement ok
delete_file(jsonl_01) # temporaire