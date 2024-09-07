import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import warnings
import json
import re
import sys

pd.set_option('display.max_rows', 100)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
pd.set_option('future.no_silent_downcasting', True)

def count_chunks(file_path, chunk_size):
    with open(file_path, 'r') as file:
        line_count = sum(1 for _ in file)
    estimated_chunks = (line_count + chunk_size - 1) // chunk_size
    print(f"total chunk estimated: {estimated_chunks}")
    return estimated_chunks

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"file deleted: {file_path}")
    else:
        print(f"ERROR, does not exists: {file_path}")

def ecoscore_score_processing(df, median_ecoscore_score): 
    df['ecoscore_score'] = pd.to_numeric(df['ecoscore_score'], errors='coerce')
    df['ecoscore_score'] = df['ecoscore_score'].apply(lambda x: max(0, min(x, 100)) if pd.notna(x) else x)
    df['ecoscore_score'] = df['ecoscore_score'].fillna(median_ecoscore_score)
    return df

def countries_processing(df, median_countries): 
    df['countries'] = pd.to_numeric(df['countries'], errors='coerce')
    df['countries'] = df['countries'].fillna(median_countries)
    return df

def groups_processing(df, median_groups): 
    df['groups'] = pd.to_numeric(df['groups'], errors='coerce')
    df['groups'] = df['groups'].fillna(median_groups)
    return df

def process_chunk_test_train(chunk, median_countries, median_ecoscore_score, median_groups):
    df = chunk.copy()
    df = ecoscore_score_processing(df, median_ecoscore_score)
    df = countries_processing(df, median_countries)
    df = groups_processing(df, median_groups)
    df = df[df['ecoscore_tags'] != 'not-applicable']
    return df

def remove_percent_nan(df, column):
    nan_rows = df[df[column].isna()]
    keep_nan_rows = nan_rows.sample(frac=0.05, random_state=42)
    non_nan_rows = df[df[column].notna()]
    df_cleaned = pd.concat([non_nan_rows, keep_nan_rows])
    return df_cleaned.reset_index(drop=True)

# lecture et traitement du fichier jsonl en morceaux train test
def browse_file_test_train(estimated_chunks, input_file, output_file, chunk_size, median_countries, median_ecoscore_score, median_groups):
    chunk_iter = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for chunk in pd.read_json(infile, lines=True, chunksize=chunk_size):
            chunk_iter +=1
            chunk = remove_percent_nan(chunk, 'ecoscore_score')
            processed_chunk = process_chunk_test_train(chunk, median_countries, median_ecoscore_score, median_groups)
            processed_chunk.to_json(outfile, orient='records', lines=True)
            print(f"-----------------------------------------------------------> progress: {(chunk_iter * 100) / estimated_chunks} %")            

def process_chunk_valid(chunk, median_countries, median_groups):
    df = chunk.copy()
    df = countries_processing(df, median_countries)
    df = groups_processing(df, median_groups)
    df = df[df['ecoscore_tags'] != 'not-applicable']
    return df

# lecture et traitement du fichier jsonl en morceaux valid
def browse_file_valid(estimated_chunks, input_file, output_file, chunk_size, median_countries, median_groups):
    chunk_iter = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for chunk in pd.read_json(infile, lines=True, chunksize=chunk_size):
            chunk_iter +=1
            processed_chunk = process_chunk_valid(chunk, median_countries, median_groups)
            processed_chunk.to_json(outfile, orient='records', lines=True)
            print(f"-----------------------------------------------------------> progress: {(chunk_iter * 100) / estimated_chunks} %")       

# utilise fichier de validation pour calculer mediane ecoscore 
def calculate_global_median(file_path, column_name, chunksize):
    all_values = []
    for chunk in pd.read_json(file_path, lines=True, chunksize=chunksize):
        if column_name in chunk.columns:
            all_values.extend(chunk[column_name].dropna().tolist())
    if all_values:
        return np.median(all_values)
    else:
        raise ValueError(f"Error, only empty values {column_name}")
    


###############################################################################
# MAIN ########################################################################
###############################################################################
def main(chunk_size, file_id, data_path):
    chunk_size = int(chunk_size)
    valid = data_path + file_id + '_valid.jsonl' 
    median_ecoscore_score = calculate_global_median(valid, 'ecoscore_score', chunk_size)
    print(f"median in validation file for ecoscore score: {median_ecoscore_score}")
    median_groups = calculate_global_median(valid, 'groups', chunk_size)
    print(f"median in validation file for groups: {median_groups}")
    median_countries = calculate_global_median(valid, 'countries', chunk_size)
    print(f"median in validation file for countries: {median_countries}")

    print("TRAIN")
    train = data_path + file_id + '_train.jsonl' 
    train_01 = data_path + file_id + '_train_01.jsonl' 
    print("estimating necessary chunk number train")
    estimated_chunks_train = count_chunks(train, chunk_size)
    print("browse throw train file to process columns")
    browse_file_test_train(estimated_chunks_train, train, train_01, chunk_size, median_countries, median_ecoscore_score, median_groups)

    print("TEST")
    test = data_path + file_id + '_test.jsonl' 
    test_01 = data_path + file_id + '_test_01.jsonl' 
    print("estimating necessary chunk number test")
    estimated_chunks_test = count_chunks(test, chunk_size)
    print("browse throw test file to process columns")
    browse_file_test_train(estimated_chunks_test, test, test_01, chunk_size, median_countries, median_ecoscore_score, median_groups)

    print("VALIDATION")
    valid = data_path + file_id + '_valid.jsonl' 
    valid_01 = data_path + file_id + '_valid_01.jsonl' 
    print("estimating necessary chunk number valid")
    estimated_chunks_valid = count_chunks(valid, chunk_size)
    print("browse throw valid file to process columns")
    browse_file_valid(estimated_chunks_valid, valid, valid_01, chunk_size, median_countries, median_groups)
    
    print("deleting file jsonl train 00")
    delete_file(train)
    print("deleting file jsonl test 00")
    delete_file(test)
    print("deleting file jsonl valid 00")
    delete_file(valid)

if __name__ == "__main__":
    chunk_size = sys.argv[1]
    file_id = sys.argv[2]
    data_path = sys.argv[3]
    main(chunk_size, file_id, data_path)