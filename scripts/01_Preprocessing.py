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

pd.set_option('display.max_rows', 50)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import jsonlines
import dask


print(dask.__version__)


import json

# Fichier d'entrée et de sortie
file_path = '/home/carolus/Documents/school/green_ia/data/04_openfoodfacts.jsonl'
output_file_path = '/home/carolus/Documents/school/green_ia/data/filtered_openfoodfacts.jsonl'

# Colonnes à garder
col_to_keep = ['allergens_from_ingredients',
               'nutriscore_tags',
               'labels_old',
               'categories_old',
               'pnns_groups_1',
               'ecoscore_data',
               'brand_owner_imported',
               'ingredients_tags',
               'packaging',
               'ingredients_hierarchy',
               'product_name',
               'food_groups_tags',
               'ecoscore_tags',
               'nova_group',
               'ingredients_from_or_that_may_be_from_palm_oil_n',
               'categories_tags',
               'brand_owner',
               'nutrient_levels_tags',
               'allergens_tags',
               'ecoscore_extended_data',
               'categories',
               'nutriments',
               'nutriscore_2021_tags',
               'additives_old_n',
               'ecoscore_score',
               'labels_tags',
               'countries']

# Traitement du fichier par petits morceaux
chunk_size = 1000  # Nombre de lignes par chunk

with open(file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    buffer = []
    
    for i, line in enumerate(infile):
        # Chargement de chaque ligne comme un dictionnaire
        record = json.loads(line.strip())
        
        # Sélection des colonnes souhaitées
        filtered_record = {key: record.get(key) for key in col_to_keep}
        
        # Ajout au buffer
        buffer.append(json.dumps(filtered_record) + '\n')
        
        # Écriture du buffer dans le fichier de sortie lorsque la taille du chunk est atteinte
        if len(buffer) >= chunk_size:
            outfile.writelines(buffer)
            buffer = []
    
    # Écriture du reste du buffer
    if buffer:
        outfile.writelines(buffer)




