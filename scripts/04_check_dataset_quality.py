import numpy as np
import pandas as pd
import os
import warnings
from datetime import datetime 
import json
import re
from langdetect import detect
from googletrans import Translator
import random


pd.set_option('display.max_rows', 50)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


# configuration
file_id = '02'
project_path = "/home/carolus/Documents/school/green_ia/" 
train = project_path + 'data/' + file_id + '_train_03.jsonl' 
test = project_path + 'data/' + file_id + '_test_03.jsonl' 
valid = project_path + 'data/' + file_id + '_valid_03.jsonl' 


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

def check_split_files(train_file, test_file, valid_file):
    def count_lines_with_condition(file_path, condition):
        count = 0
        total_lines = 0
        with open(file_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                total_lines += 1
                if condition(record):
                    count += 1
        return count, total_lines

    def print_stats(label, count, total_lines):
        percentage = (count / total_lines) * 100 if total_lines > 0 else 0
        print(f"{label}: {count} lignes ({percentage:.2f}%)")

    def check_file(file_path, valid_low_count, test_low_count, train_low_count, valid_other_count, test_other_count, train_other_count):
        valid_low, total_valid = count_lines_with_condition(file_path, lambda x: x.get('ecoscore_note', 0) < 101)
        valid_other, total_valid_other = count_lines_with_condition(file_path, lambda x: x.get('ecoscore_note', 0) >= 101)
        
        if total_valid != valid_low or total_valid_other != valid_other:
            print(f"Erreur : La répartition des lignes avec ecoscore_note < 101 dans {file_path} est incorrecte.")
            return False
        
        print_stats("Lignes avec ecoscore_note < 101", valid_low, total_valid)
        print_stats("Lignes avec ecoscore_note >= 101", valid_other, total_valid_other)
        
        total_lines = valid_low + valid_other
        if not (abs(valid_low - valid_low_count) < 0.01 * valid_low_count and
                abs(valid_other - valid_other_count) < 0.01 * valid_other_count):
            print(f"Erreur : Les quantités de lignes dans {file_path} ne correspondent pas aux attentes.")
            return False
        
        return True

    def print_summary():
        train_total = sum([train_low_count, train_other_count])
        test_total = sum([test_low_count, test_other_count])
        valid_total = sum([valid_low_count, valid_other_count])

        print(f"\nVérification des fichiers:")
        print(f"  Fichier Train: {train_total} lignes")
        print(f"  Fichier Test: {test_total} lignes")
        print(f"  Fichier Validation: {valid_total} lignes")
    
    # Déterminer les tailles attendues pour chaque fichier
    num_low_ecoscore = sum(count_lines_with_condition(file, lambda x: x.get('ecoscore_note', 0) < 101)[0] for file in [train_file, test_file, valid_file])
    num_other = sum(count_lines_with_condition(file, lambda x: x.get('ecoscore_note', 0) >= 101)[0] for file in [train_file, test_file, valid_file])
    
    valid_low_count = int(0.05 * num_low_ecoscore)
    test_low_count = int(0.15 * num_low_ecoscore)
    train_low_count = num_low_ecoscore - valid_low_count - test_low_count

    valid_other_count = int(0.05 * num_other)
    test_other_count = int(0.15 * num_other)
    train_other_count = num_other - valid_other_count - test_other_count

    # Vérifier chaque fichier
    if check_file(train_file, valid_low_count, test_low_count, train_low_count, valid_other_count, test_other_count, train_other_count):
        print("Fichier Train vérifié avec succès.")
    if check_file(test_file, valid_low_count, test_low_count, train_low_count, valid_other_count, test_other_count, train_other_count):
        print("Fichier Test vérifié avec succès.")
    if check_file(valid_file, valid_low_count, test_low_count, train_low_count, valid_other_count, test_other_count, train_other_count):
        print("Fichier Validation vérifié avec succès.")
    
    # Afficher le résumé
    print_summary()


check_split_files(train, test, valid)



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