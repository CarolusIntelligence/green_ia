import numpy as np
import pandas as pd
import jsonlines
import os
import warnings
from datetime import datetime 
import json
import re
from langdetect import detect
from googletrans import Translator
import random
import sys
import math

pd.set_option('display.max_rows', 50)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

def main(chunk_size, file_id, project_path):
    print("\n\n01_preprocessing \n")
    print("chunk_size:", chunk_size)
    print("File ID:", file_id)
    print("Project Path:", project_path)

    # configuration
    jsonl_00 = project_path + "data/" + file_id + "_openfoodfacts_00" + ".jsonl" # fichier sans aucune étape de prétraitement (dézipé) 
    jsonl_01 = project_path + 'data/' + file_id + '_openfoodfacts_01.jsonl' # fichier avec première étape de prétraitement (uniquement colonnes intéressantes)
    jsonl_02 = project_path + 'data/' + file_id + '_openfoodfacts_02.jsonl' # fichier avec deuxième étape de prétraitement (traitement intégral)
    jsonl_03 = project_path + 'data/' + file_id + '_openfoodfacts_03.jsonl' # fichier avec troisième étape de prétraitement, mélange des lignes aléatoirement
    train = project_path + "data/" + file_id + "_train" + ".jsonl"
    test = project_path + "data/" + file_id + "_test" + ".jsonl"
    valid = project_path + "data/" + file_id + "_valid" + ".jsonl"
    jsonl_sample = project_path + 'data/' + file_id + '_openfoodfacts_sample.jsonl'
    col_to_keep = ['pnns_groups_1',
                'ingredients_tags',
                'packaging',
                'product_name',
                'ecoscore_tags',
                'categories_tags',
                'ecoscore_score',
                'labels_tags',
                'code']


    # récupérer la date du jour 
    current_date_time = datetime.now()
    date_format = "%d/%m/%Y %H:%M:%S.%f"
    start_date = current_date_time.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
    date_code = current_date_time.strftime('%d%m%Y%H%M%S') + f"{current_date_time.microsecond // 1000:03d}"


    def add_logs(logData):
        print(logData)
        with open(f"{project_path}logs/01_preprocessing_{date_code}_logs.txt", "a") as logFile:
            logFile.write(f'{logData}\n')

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
            add_logs(f"file deleted: {file_path}")
        else:
            add_logs(f"ERROR, does not exists: {file_path}")


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
        add_logs(f"jsonl generated, 01: {origin_file}")


    # création d'un échantillion (mini jsonl) pour inspecter la qualité des données
    def jsonl_sample_creator(file_to_sample, jsonl_sample, num_samples=60):
        add_logs(f"sampling {num_samples} random lines from {file_to_sample} to {jsonl_sample}")
        with open(file_to_sample, 'r') as infile:
            total_lines = sum(1 for _ in infile)
        add_logs(f"total number of lines jsonl, 02: {total_lines}")
        sample_indices = random.sample(range(total_lines), num_samples)
        with open(file_to_sample, 'r') as infile, open(jsonl_sample, 'w') as outfile:
            for current_line_number, line in enumerate(infile):
                if current_line_number in sample_indices:
                    outfile.write(line)
        add_logs(f"jsonl sample created: {jsonl_sample}")


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
            df['groups'] = df['groups'].str.lower() 
            #df['groups'] = df['groups'].apply(translate_to_english)


            # traitement col NAME
            df['name'] = df['name'].replace("", None)  
            df['name'] = df['name'].replace({np.nan: None})
            df['name'] = df['name'].str.lower()
            #df['name'] = df['name'].apply(translate_to_english)


            # traitement col CODE
            df['code'] = df['code'].replace("", None)  
            df['code'] = df['code'].replace({np.nan: None})
            df['code'] = pd.to_numeric(df['code'], errors='coerce')
            df['code'] = df['code'].apply(lambda x: np.nan if pd.isna(x) else int(round(x)))


            # supprime les lignes où le code unique ou le nom produit sont absents 
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
            df['packaging'] = df['packaging'].astype(str)
            df['packaging'] = df['packaging'].str.lower()
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


            # traitment col ECOSCORE_NOTE
            df['ecoscore_note'] = df['ecoscore_note'].replace("unknown", 999)
            df['ecoscore_note'] = df['ecoscore_note'].replace("", 999)
            df['ecoscore_note'] = df['ecoscore_note'].fillna(999)
            # remplace toutes les valeurs < 0 par 0, et toutes celles > 100 par 100
            df['ecoscore_note'] = df['ecoscore_note'].apply(lambda x: max(0, min(x, 100)) if x < 999 else x)


            # supprime les lignes avec trop de None
            df = df[~(
            (df['groups'].isna() & df['categories'].isna()) |
            (df['ecoscore_groups'].isna() & df['groups'].isna()) |
            (df['ecoscore_groups'].isna() & df['categories'].isna())
            )]


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
            # ramène toutes les notes > 9 à 9
            df['labels_note'] = df['labels_note'].apply(lambda x: min(x, 9) if pd.notna(x) else x)
            return df 



        # lecture et traitement du fichier jsonl en morceaux
        estimated_chunks = count_chunks(jsonl_01, chunk_size)
        chunk_iter = 0
        add_logs(f"start time preprocessing : {get_time()}, total chunk estimated: {estimated_chunks}")
        with open(jsonl_01, 'r') as infile, open(jsonl_02, 'w') as outfile:
            for chunk in pd.read_json(infile, lines=True, chunksize=chunk_size):
                chunk_iter = chunk_iter + 1
                processed_chunk = process_chunk(chunk)
                processed_chunk.to_json(outfile, orient='records', lines=True)
                add_logs(f"saved content, time: {get_time()}, progress: {(chunk_iter * 100) / estimated_chunks}%")




    def read_in_chunks(file_path, chunk_size):
        with open(file_path, 'r', encoding='utf-8') as f:
            chunk = []
            for line in f:
                chunk.append(json.loads(line))
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk

    def shuffle_jsonl(jsonl_02, jsonl_03, chunk_size):
        temp_file = jsonl_03 + '.temp'
        with open(temp_file, 'w', encoding='utf-8') as temp_f:
            for chunk in read_in_chunks(jsonl_02, chunk_size):
                random.shuffle(chunk)
                for obj in chunk:
                    temp_f.write(json.dumps(obj) + '\n')
        with open(temp_file, 'r', encoding='utf-8') as temp_f:
            lines = temp_f.readlines()
            random.shuffle(lines)
        with open(jsonl_03, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        os.remove(temp_file)


    def line_count(jsonl_03, type):
        count = 0
        with open(jsonl_03, 'r', encoding='utf-8') as file:
            for line in file:
                if (type == 0):
                    try:
                        obj = json.loads(line)
                        if 'ecoscore_note' in obj:
                            value = obj['ecoscore_note']
                            if isinstance(value, (int, float)) and value == 999:
                                count += 1
                    except json.JSONDecodeError:
                        print("Erreur de décodage JSON dans la ligne suivante :")
                        print(line)
                        continue

                elif (type == 1):
                    try:
                        obj = json.loads(line)
                        if 'ecoscore_note' in obj:
                            value = obj['ecoscore_note']
                            if isinstance(value, (int, float)) and 0 <= value <= 999:
                                count += 1
                    except json.JSONDecodeError:
                        print("Erreur de décodage JSON dans la ligne suivante :")
                        print(line)
                        continue

                elif (type == 2):
                    try:
                        obj = json.loads(line)
                        if 'ecoscore_note' in obj:
                            value = obj['ecoscore_note']
                            if isinstance(value, (int, float)) and 0 <= value <= 100:
                                count += 1
                    except json.JSONDecodeError:
                        print("Erreur de décodage JSON dans la ligne suivante :")
                        print(line)
                        continue
        return count

    def line_repartitor(jsonl_03, train, test, valid, train_nb_line_ko, train_nb_line_ok, test_nb_line_ko, test_nb_line_ok, valid_nb_line_ko, valid_nb_line_ok):
        with jsonlines.open(train, mode='w') as train_writer, \
            jsonlines.open(test, mode='w') as test_writer, \
            jsonlines.open(valid, mode='w') as valid_writer:
            
            train_ok_iter, train_ko_iter = 0, 0
            test_ok_iter, test_ko_iter = 0, 0
            valid_ok_iter, valid_ko_iter = 0, 0
            total_iter, ok_iter, ko_iter = 0, 0, 0
            
            with jsonlines.open(jsonl_03, mode='r') as reader:
                for obj in reader:
                    ecoscore_note = obj.get('ecoscore_note', float('inf'))
                    total_iter+=1

                    if (ecoscore_note == 999):
                        if (valid_ko_iter < valid_nb_line_ko):
                            valid_writer.write(obj)
                            valid_ko_iter+=1
                        elif (test_ko_iter < test_nb_line_ko):
                            test_writer.write(obj)
                            test_ko_iter+=1
                        elif (train_ko_iter < train_nb_line_ko):
                            train_writer.write(obj)
                            train_ko_iter+=1    
                        ko_iter+=1

                    elif(ecoscore_note < 101 and ecoscore_note >= 0):                    
                        if (valid_ok_iter < valid_nb_line_ok):
                            valid_writer.write(obj)
                            valid_ok_iter+=1
                        elif (test_ok_iter < test_nb_line_ok):
                            test_writer.write(obj)
                            test_ok_iter+=1
                        elif (train_ok_iter < train_nb_line_ok):
                            train_writer.write(obj)
                            train_ok_iter+=1    
                        ok_iter+=1

            add_logs(f"nombre objets comptés: {total_iter}")
            add_logs(f"ecoscore ok comptés: {ok_iter}")
            add_logs(f"ecoscore ko comptés: {ko_iter}")
            add_logs(f"lignes ko ajoutés à valid: {valid_ko_iter}")
            add_logs(f"lignes ko ajoutés à test: {test_ko_iter}")
            add_logs(f"lignes ko ajoutés à train: {train_ko_iter}")
            add_logs(f"lignes ok ajoutés à valid: {valid_ok_iter}")
            add_logs(f"lignes ok ajoutés à test: {test_ok_iter}")
            add_logs(f"lignes ok ajoutés à train: {train_ok_iter}")
                    
    def split_jsonl_file(jsonl_02, train, test, valid, jsonl_03, chunk_size):
        shuffle_jsonl(jsonl_02, jsonl_03, chunk_size) # mélanger toutes les lignes aléatoirement dans jsonl_02
        valid_ecoscore_count = line_count(jsonl_03, type = 2) # compter le nombre de lignes avec écoscore 
        invalid_ecoscore_count = line_count(jsonl_03, type = 0) # compter le nombre de lignes autres (sans écoscore)
        line_count_number = line_count(jsonl_03, type = 1) # compter le nombre de lignes total
        # compter le nombre de lignes pour chaque fichier 
        train_nb_line_ko = math.floor((invalid_ecoscore_count * 80) / 100) # train ecoscore ko
        train_nb_line_ok = math.floor((valid_ecoscore_count * 80) / 100) # train ecoscore ok
        test_nb_line_ko = math.floor((invalid_ecoscore_count * 20) / 100) # test ecoscore ko
        test_nb_line_ok = math.floor((valid_ecoscore_count * 15) / 100) # test ecoscore ok
        valid_nb_line_ko = math.floor((invalid_ecoscore_count * 0) / 100) # valid ecoscore ko
        valid_nb_line_ok = math.floor((valid_ecoscore_count * 5) / 100) # valid ecoscore ok 
        add_logs(f"ecoscore ok: {valid_ecoscore_count}")
        add_logs(f"ecoscore ko: {invalid_ecoscore_count}")
        add_logs(f"nombre d'objets total: {line_count_number}")
        add_logs(f"ko attendus dans train: {train_nb_line_ko}")
        add_logs(f"ok attendus dans train: {train_nb_line_ok}")
        add_logs(f"ko attendus dans test: {test_nb_line_ko}")
        add_logs(f"ok attendus dans test: {test_nb_line_ok}")
        add_logs(f"ko attendus dans valid: {valid_nb_line_ko}")
        add_logs(f"ok attendus dans valid: {valid_nb_line_ok}")
        # répartir les lignes entre les fichiers
        line_repartitor(jsonl_03, train, test, valid, train_nb_line_ko, train_nb_line_ok, test_nb_line_ko, test_nb_line_ok, valid_nb_line_ko, valid_nb_line_ok)

    add_logs("01_preprocessing logs:")
    add_logs(f"chunk_size: {chunk_size} \nfile_id: {file_id} \nproject_path: {project_path} \njsonl_00 {jsonl_00} \njsonl_01: {jsonl_01} \njsonl_02: {jsonl_02} \njsonl_sample: {jsonl_sample} \ncol_to_keep: {col_to_keep}, \nstart_date: {start_date}, \ntrain: {train}, \ntest: {test}, \nvalid: {valid}, \njsonl_03: {jsonl_03}")

    # main algo
    jsonl_filtered_creator(jsonl_00)
    delete_file(jsonl_00)
    main_processing(jsonl_01, jsonl_02)
    delete_file(jsonl_01)
    jsonl_sample_creator(jsonl_02, jsonl_sample) # puis utiliser 02 car prétraitement ok
    split_jsonl_file(jsonl_02, train, test, valid, jsonl_03, chunk_size)


    # récupérer la date du jour 
    current_date_time = datetime.now()
    end_date = current_date_time.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
    add_logs(f"end date: {end_date}")

    # afficher temps total execution script 
    start_date = datetime.strptime(start_date, date_format)
    end_date = datetime.strptime(end_date, date_format)
    time_difference = end_date - start_date
    time_difference_minutes = time_difference.total_seconds() / 60
    add_logs(f"execution script time: {time_difference_minutes:.2f} minutes")


if __name__ == "__main__":
    chunk_size = sys.argv[1]
    file_id = sys.argv[2]
    project_path = sys.argv[3]
    
    main(chunk_size, file_id, project_path)