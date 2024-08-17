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
import re
from langdetect import detect
from googletrans import Translator
import random


# configuration
chunk_size = 1000
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
               'code',
               'countries']

def get_time():
    current_date = datetime.now()
    current_date = current_date.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
    return current_date


def count_chunks(file_path, chunk_size):
    with open(file_path, 'r') as file:
        line_count = sum(1 for _ in file)
    total_chunks = (line_count + chunk_size - 1) // chunk_size
    return total_chunks


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
def jsonl_sample_creator(file_to_sample, jsonl_sample, num_samples=90):
    print(f"Sampling {num_samples} random lines from {file_to_sample} to {jsonl_sample}")
    with open(file_to_sample, 'r') as infile:
        total_lines = sum(1 for _ in infile)
    print(f"Total number of lines: {total_lines}")
    sample_indices = random.sample(range(total_lines), num_samples)
    with open(file_to_sample, 'r') as infile, open(jsonl_sample, 'w') as outfile:
        for current_line_number, line in enumerate(infile):
            if current_line_number in sample_indices:
                outfile.write(line)
    print(f"Sample created and saved to {jsonl_sample}")


def main_processing(jsonl_01, jsonl_02):
    # traducteur, remplace le contenu d'une autre langue que l'anglais en anglais 
    translator = Translator()
    
    def translate_to_english(text):
        if text is None:
            return text
        try:
            detected_lang = detect(text)
            if detected_lang == 'en':
                return text.lower()
            else:
                translated = translator.translate(text, dest='en')
                return translated.text.lower()
        except Exception as e:
            return text
        
    def process_chunk(chunk):
        df = chunk.copy()

        # renommer les colonnes
        df.rename(columns={'pnns_groups_1': 'groups'}, inplace=True)
        df.rename(columns={'ingredients_tags': 'ingredients_temp'}, inplace=True)
        df.rename(columns={'product_name': 'name'}, inplace=True)
        df.rename(columns={'ecoscore_tags': 'ecoscore_groups'}, inplace=True)
        df.rename(columns={'categories_tags': 'categories_temp'}, inplace=True)
        df.rename(columns={'ecoscore_score': 'ecoscore_note'}, inplace=True)
        df.rename(columns={'labels_tags': 'labels_temp'}, inplace=True)


        # traitement col GROUPS 
        df['groups'] = df['groups'].replace("unknown", None, regex=False)  
        #df['groups'] = df['groups'].apply(translate_to_english)


        # traitement col NAME
        df['name'] = df['name'].replace("", None)  
        df['name'] = df['name'].replace({np.nan: None})
        #df['name'] = df['name'].apply(translate_to_english)


        # traitement col CODE
        df['code'] = df['code'].replace("", None)  
        df['code'] = df['code'].replace({np.nan: None})
        df['code'] = pd.to_numeric(df['code'], errors='coerce')
        df['code'] = df['code'].apply(lambda x: np.nan if pd.isna(x) else int(round(x)))


        # supprime les lignes où le code ean ou le nom produit sont absents 
        df = df[df['name'].notna() & df['code'].notna()]


        # traitement col INGREDIENTS
        df['ingredients_temp'] = df['ingredients_temp'].replace("", None)  # remplace vide par None
        df['ingredients_temp'] = df['ingredients_temp'].replace({np.nan: None}) # remplace NaN par None
        df['ingredients_temp'] = df['ingredients_temp'].apply(lambda x: x if isinstance(x, list) else []) # remplace None par liste vide 
        df['ingredients_temp'] = df['ingredients_temp'].apply(lambda x: ', '.join(x)) # converti liste en string 
        # extraire éléments avec 'en:' nouvelle colonne
        def extract_en_ingredients(ingredient_list):
            ingredients = ingredient_list.strip('[]').split(', ')
            return [ingredient.split(':')[-1] for ingredient in ingredients if ingredient.startswith('en:')] 
        df['ingredients'] = df['ingredients_temp'].apply(extract_en_ingredients)
        df.drop(columns=['ingredients_temp'], inplace=True)
        df['ingredients'] = df['ingredients'].apply(lambda x: ', '.join(x))
        df['ingredients'] = df['ingredients'].replace("", None)  
        #df['ingredients'] = df['ingredients'].apply(translate_to_english)


        # traitement col PACKAGING
        df['packaging'] = df['packaging'].replace("", None)
        def remove_two_letters_and_colon(s):
            if isinstance(s, str):
                return re.sub(r'\b\w{2}:\b', '', s)
            return s
        df['packaging'] = df['packaging'].apply(remove_two_letters_and_colon)
        #df['packaging'] = df['packaging'].apply(translate_to_english)


        # traitement col ECOSCORE_GROUPS
        df['ecoscore_groups'] = df['ecoscore_groups'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x) # conversion liste vers string 
        df['ecoscore_groups'] = df['ecoscore_groups'].replace("unknown", "z")
        df['ecoscore_groups'] = df['ecoscore_groups'].replace("", "z")
        df['ecoscore_groups'] = df['ecoscore_groups'].fillna("z") 
        df['ecoscore_groups'] = df['ecoscore_groups'].replace("not-applicable", "z")


        # traitement col CATEGORIES
        df['categories_temp'] = df['categories_temp'].replace("", None)  
        df['categories_temp'] = df['categories_temp'].replace({np.nan: None}) 
        df['categories_temp'] = df['categories_temp'].apply(lambda x: x if isinstance(x, list) else [])
        df['categories_temp'] = df['categories_temp'].apply(lambda x: ', '.join(x))
        # extraire éléments avec 'en:' nouvelle colonne
        def extract_en_categories(categories_list):
            ingredients = categories_list.strip('[]').split(', ')
            return [ingredient.split(':')[-1] for ingredient in ingredients if ingredient.startswith('en:')]
        df['categories'] = df['categories_temp'].apply(extract_en_categories)
        df.drop(columns=['categories_temp'], inplace=True)
        df['categories'] = df['categories'].apply(lambda x: ', '.join(x))
        df['categories'] = df['categories'].replace("", None)  
        #df['categories'] = df['categories'].apply(translate_to_english)


        # traitment col COUNTRIES
        def clean_abrev(texte):
            if isinstance(texte, str):
                return re.sub(r'\b\w{2}:\b', '', texte).strip()
            return texte

        country_mapping = {
            'fr': 'france',
            'us': 'united states',
            'ca': 'canada',
            'ie': 'ireland',
            'it': 'italy',
            'za': 'south africa',
            'ch': 'switzerland',
            'suisse': 'switzerland',
            'gb': 'united kingdom',
            'be': 'belgium',
            'no': 'norway',
            'es': 'spain',
            'jp': 'japan', 
            'de': 'germany', 
            've': 'venezuela', 
            'au': 'australia', 
            'dz': 'algeria', 
            'ma': 'morocco', 
            'ro': 'romania', 
            'vg': 'united kingdom', 
            'pf': 'french polynesia', 
            'at': 'austria', 
            'pr': 'puerto rico', 
            'nl': "new zealand", 
            'sn': "senegal",
            'españa': 'spain', 
            'monde': 'world', 
            'gi': 'gibraltar', 
            'frankreich': 'france', 
            'sa': 'saudi arabia', 
            'tunisie': 'tunisia', 
            'polska': 'poland', 
            'србија': 'serbia', 
            'dänemark': 'denmark', 
            'mt': 'malte', 
            'lu': 'luxembourg', 
            'nederland': 'netherlands', 
            'lb': 'lebanon', 
            'ly': 'lybia', 
            're': 'reunion',
            'frankreich': 'france'
        }
        df['countries'] = df['countries'].replace("", None)  # Remplacer les chaînes vides par None
        df['countries'] = df['countries'].apply(lambda x: x if isinstance(x, list) else ([] if x is None else x.split(', ')))
        df['countries'] = df['countries'].apply(lambda x: ', '.join(x) if x else None)
        df['countries'] = df['countries'].str.lower()  # Convertir en minuscule
        df['countries'] = df['countries'].apply(clean_abrev)  # Appliquer la fonction de nettoyage
        df['countries'] = df['countries'].replace(country_mapping)  # Remplacer les acronymes grâce au dictionnaire
        df['countries'] = df['countries'].fillna('None')  # Remplacer les None par 'None'


        # traitment col ECOSCORE_NOTE
        df['ecoscore_note'] = df['ecoscore_note'].replace("unknown", 999)
        df['ecoscore_note'] = df['ecoscore_note'].replace("", 999)
        df['ecoscore_note'] = df['ecoscore_note'].fillna(999)


        # traitment col LABELS
        df['labels_temp'] = df['labels_temp'].replace("", None)
        df['labels_temp'] = df['labels_temp'].replace({np.nan: None})
        df['labels_temp'] = df['labels_temp'].apply(lambda x: x if isinstance(x, list) else ([] if x is None else x.split(', ')))
        def extract_en_labels(labels_list):
            if isinstance(labels_list, str):
                labels_list = labels_list.split(', ')
            return [ingredient.split(':', 1)[-1] for ingredient in labels_list if ingredient.startswith('en:')]

        df['labels'] = df['labels_temp'].apply(extract_en_labels)
        df['labels'] = df['labels'].apply(lambda x: ', '.join(x) if x else None)
        df.drop(columns=['labels_temp'], inplace=True)

        def count_commas_plus_one(value):
            if pd.isna(value):  
                return 0
            return value.count(',') + 1
        df['labels_note'] = df['labels'].apply(count_commas_plus_one)
        df.drop(columns=['labels'], inplace=True)

        return df 


    # lecture et traitement du fichier jsonl en morceaux
    print(f"start time: {get_time()}, total chunk estimated: {count_chunks(jsonl_01, chunk_size)}")
    processed_chunks = 0
    with open(jsonl_01, 'r') as infile, open(jsonl_02, 'w') as outfile:
        for chunk in pd.read_json(infile, lines=True, chunksize=chunk_size):
            processed_chunk = process_chunk(chunk)
            processed_chunk.to_json(outfile, orient='records', lines=True)
            print(f"saved content, time: {get_time()}")


# main algo
#jsonl_filtered_creator(jsonl_00)
#delete_file(jsonl_00)
main_processing(jsonl_01, jsonl_02)
#delete_file(jsonl_01)
jsonl_sample_creator(jsonl_02, jsonl_sample) # puis utiliser 02 car prétraitement ok