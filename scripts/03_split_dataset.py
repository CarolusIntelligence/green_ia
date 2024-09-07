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

pd.set_option('display.max_rows', 50)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"file deleted: {file_path}")
    else:
        print(f"ERROR, does not exists: {file_path}")

def line_count(jsonl_03, type):
    count = 0
    with open(jsonl_03, 'r', encoding='utf-8') as file:
        for line in file:
            if (type == 0): # sans écoscore score 
                try:
                    obj = json.loads(line)
                    if 'ecoscore_score' in obj:
                        value = obj['ecoscore_score']
                        #if isinstance(value, (int, float)) and value == 999:
                        if value is None or (isinstance(value, (int, float)) and np.isnan(value)):
                            count += 1
                except json.JSONDecodeError:
                    print("Erreur de décodage JSON dans la ligne suivante :")
                    print(line)
                    continue
            elif (type == 1): # compte toutes les lignes 
                try:
                    obj = json.loads(line)
                    if 'ecoscore_score' in obj:
                        value = obj['ecoscore_score']
                        if value is None or np.isnan(value) or (isinstance(value, (int, float)) and 0 <= value <= 100):
                            count += 1
                except json.JSONDecodeError:
                    print("Erreur de décodage JSON dans la ligne suivante :")
                    print(line)
                    continue
            elif (type == 2): # lignes avec écoscore score 
                try:
                    obj = json.loads(line)
                    if 'ecoscore_score' in obj:
                        value = obj['ecoscore_score']
                        if isinstance(value, (int, float)) and not math.isnan(value) and not None:
                            count += 1
                except json.JSONDecodeError:
                    print("Erreur de décodage JSON dans la ligne suivante :")
                    print(line)
                    continue
    return count

def validation(total_iter, ok_iter, ko_iter, valid_ko_iter, test_ko_iter, train_ko_iter, valid_ok_iter, test_ok_iter, train_ok_iter): 
    ok_check, ko_check, count_check = False, False, False
    # count
    check_ok_count = (100 * ok_iter) / total_iter
    check_ko_count = (100 * ko_iter) / total_iter
    check_count_sum = check_ko_count + check_ok_count
    if (check_count_sum > 99 and check_count_sum < 101):
        print(f"count ko/ok valid, 99 < {check_count_sum} < 101")
        count_check = True
    else:
        print(f"ERROR, count ko/ok invalid, 99 < {check_count_sum} < 101")
        count_check = False
    # validation, train, test ko
    check_test_ko = (100 * test_ko_iter) / ko_iter
    check_valid_ko = (100 * valid_ko_iter) / ko_iter
    check_train_ko = (100 * train_ko_iter) / ko_iter
    check_datasets_sum_ko = check_valid_ko + check_train_ko + check_test_ko
    if (check_datasets_sum_ko > 99 and check_datasets_sum_ko < 101):
        print(f"datasets repartition ko valid, 99 < {check_datasets_sum_ko} < 101")
        ko_check = True
    else:
        print(f"ERROR, datasets repartition ko invalid, 99 < {check_datasets_sum_ko} < 101")
        ko_check = False
    # validation, train, test ok
    check_test_ok = (100 * test_ok_iter) / ok_iter
    check_valid_ok = (100 * valid_ok_iter) / ok_iter
    check_train_ok = (100 * train_ok_iter) / ok_iter
    check_datasets_sum_ok = check_valid_ok + check_train_ok + check_test_ok
    if (check_datasets_sum_ok > 99 and check_datasets_sum_ok < 101):
        print(f"datasets repartition ok valid, 99 < {check_datasets_sum_ok} < 101")
        ok_check = True
    else:
        print(f"ERROR, datasets repartition ok invalid, 99 < {check_datasets_sum_ok} < 101")
        ok_check = False 
    return ok_check, ko_check, count_check

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
                #ecoscore_score = obj.get('ecoscore_score', float('inf'))
                ecoscore_score = obj.get('ecoscore_score', float('nan'))
                total_iter+=1
                if (ecoscore_score is np.nan or ecoscore_score is None):
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
                elif(ecoscore_score is not np.nan or ecoscore_score is not None):                    
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
        ok_check, ko_check, count_check = validation(total_iter, ok_iter, ko_iter, valid_ko_iter, test_ko_iter, train_ko_iter, valid_ok_iter, test_ok_iter, train_ok_iter)
        print(f"ok_check: {ok_check}, ko_check: {ko_check}, count_check: {count_check}")

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

def split_jsonl_file(jsonl_03, train, test, valid, jsonl_04, chunk_size):
    shuffle_jsonl(jsonl_03, jsonl_04, chunk_size) # mélanger toutes les lignes aléatoirement dans jsonl_02
    valid_ecoscore_count = line_count(jsonl_04, type = 2) # compter le nombre de lignes avec écoscore 
    invalid_ecoscore_count = line_count(jsonl_04, type = 0) # compter le nombre de lignes autres (sans écoscore)
    line_count_number = line_count(jsonl_04, type = 1) # compter le nombre de lignes total
    # compter le nombre de lignes pour chaque fichier 
    train_nb_line_ko = math.floor((invalid_ecoscore_count * 80) / 100) # train ecoscore ko
    train_nb_line_ok = math.floor((valid_ecoscore_count * 84.9) / 100) # train ecoscore ok
    test_nb_line_ko = math.floor((invalid_ecoscore_count * 20) / 100) # test ecoscore ko
    test_nb_line_ok = math.floor((valid_ecoscore_count * 15) / 100) # test ecoscore ok
    valid_nb_line_ko = math.floor((invalid_ecoscore_count * 0) / 100) # valid ecoscore ko
    valid_nb_line_ok = math.floor((valid_ecoscore_count * 0.1) / 100) # valid ecoscore ok 
    # répartir les lignes entre les fichiers
    line_repartitor(jsonl_04, train, test, valid, train_nb_line_ko, train_nb_line_ok, test_nb_line_ko, test_nb_line_ok, valid_nb_line_ko, valid_nb_line_ok)



###############################################################################
# MAIN ########################################################################
###############################################################################
def main(chunk_size, file_id, data_path):
    chunk_size = int(chunk_size)
    jsonl_03 = data_path + file_id + '_openfoodfacts_03.jsonl' 
    jsonl_04 = data_path + file_id + '_openfoodfacts_04.jsonl' 
    train = data_path + file_id + "_train" + ".jsonl"
    test = data_path + file_id + "_test" + ".jsonl"
    valid = data_path + file_id + "_valid" + ".jsonl"
    print("start spliting dataset")
    split_jsonl_file(jsonl_03, train, test, valid, jsonl_04, chunk_size)
    #print("deleting file jsonl 03")
    #delete_file(jsonl_03)

if __name__ == "__main__":
    chunk_size = sys.argv[1]
    file_id = sys.argv[2]
    data_path = sys.argv[3]
    main(chunk_size, file_id, data_path)