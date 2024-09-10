import numpy as np
import pandas as pd
import os
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

def ecoscore_score_processing(df): 
    df['ecoscore_score'] = pd.to_numeric(df['ecoscore_score'], errors='coerce')
    df['ecoscore_score'] = df['ecoscore_score'].apply(lambda x: max(0, min(x, 100)) if pd.notna(x) else x)
    nan_indices = df[df['ecoscore_score'].isna()].index
    n_nan = len(nan_indices)
    n_to_drop = int(0.90 * n_nan)
    indices_to_drop = np.random.choice(nan_indices, size=n_to_drop, replace=False)
    df = df.drop(indices_to_drop)
    return df

def ecoscore_tags_processing(df):
    def replace_nan_based_on_score(row):
        if pd.isna(row['ecoscore_tags']):
            if 80 <= row['ecoscore_score'] <= 100:
                return 0
            elif 60 <= row['ecoscore_score'] < 80:
                return 1
            elif 40 <= row['ecoscore_score'] < 60:
                return 2
            elif 20 <= row['ecoscore_score'] < 40:
                return 3
            elif 0 <= row['ecoscore_score'] < 20:
                return 4
        return row['ecoscore_tags']
    df['ecoscore_tags'] = df.apply(replace_nan_based_on_score, axis=1)
    return df

def countries_processing(df, median_countries): 
    df['countries'] = pd.to_numeric(df['countries'], errors='coerce')
    df['countries'] = df['countries'].fillna(median_countries)
    return df

def pnns_1_processing(df, median_pnns_1): 
    df['pnns_1'] = pd.to_numeric(df['pnns_1'], errors='coerce')
    df['pnns_1'] = df['pnns_1'].fillna(median_pnns_1)
    return df

def nova_processing(df, median_nova): 
    df['nova'] = pd.to_numeric(df['nova'], errors='coerce')
    df['nova'] = df['nova'].fillna(median_nova)
    return df

def palm_oil_processing(df, median_palm_oil): 
    df['palm_oil'] = pd.to_numeric(df['palm_oil'], errors='coerce')
    df['palm_oil'] = df['palm_oil'].fillna(median_palm_oil)
    return df

def nutriscore_tags_processing(df, median_nutriscore_tags): 
    df['nutriscore_tags'] = pd.to_numeric(df['nutriscore_tags'], errors='coerce')
    df['nutriscore_tags'] = df['nutriscore_tags'].fillna(median_nutriscore_tags)
    return df

def additives_processing(df, median_additives): 
    df['additives'] = pd.to_numeric(df['additives'], errors='coerce')
    df['additives'] = df['additives'].fillna(median_additives)
    return df

def process_chunk_test_train(chunk, median_countries, median_pnns_1, median_nova, median_palm_oil, median_nutriscore_tags, median_additives):
    df = chunk.copy()
    df = ecoscore_score_processing(df)
    df = countries_processing(df, median_countries)
    df = pnns_1_processing(df, median_pnns_1)
    df = nova_processing(df, median_nova)
    df = palm_oil_processing(df, median_palm_oil)
    df = nutriscore_tags_processing(df, median_nutriscore_tags)
    df = additives_processing(df, median_additives)
    df = df[df['ecoscore_tags'] != 'not-applicable']
    df = ecoscore_tags_processing(df)
    return df

# lecture et traitement du fichier jsonl en morceaux train test
def browse_file_test_train(estimated_chunks, input_file, output_file, chunk_size, median_countries, median_pnns_1, median_nova, median_palm_oil, median_nutriscore_tags, median_additives):
    chunk_iter = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for chunk in pd.read_json(infile, lines=True, chunksize=chunk_size):
            chunk_iter +=1
            processed_chunk = process_chunk_test_train(chunk, median_countries, median_pnns_1, median_nova, median_palm_oil, median_nutriscore_tags, median_additives)
            processed_chunk.to_json(outfile, orient='records', lines=True)
            print(f"-----------------------------------------------------------> progress: {(chunk_iter * 100) / estimated_chunks} %")            

def process_chunk_valid(chunk, median_countries, median_pnns_1, median_nova, median_palm_oil, median_nutriscore_tags, median_additives):
    df = chunk.copy()
    df = countries_processing(df, median_countries)
    df = pnns_1_processing(df, median_pnns_1)
    df = nova_processing(df, median_nova)
    df = palm_oil_processing(df, median_palm_oil)
    df = nutriscore_tags_processing(df, median_nutriscore_tags)
    df = additives_processing(df, median_additives)
    return df

# lecture et traitement du fichier jsonl en morceaux valid
def browse_file_valid(estimated_chunks, input_file, output_file, chunk_size, median_countries, median_pnns_1, median_nova, median_palm_oil, median_nutriscore_tags, median_additives):
    chunk_iter = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for chunk in pd.read_json(infile, lines=True, chunksize=chunk_size):
            chunk_iter +=1
            processed_chunk = process_chunk_valid(chunk, median_countries, median_pnns_1, median_nova, median_palm_oil, median_nutriscore_tags, median_additives)
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
    train = data_path + file_id + '_train.jsonl' 
    median_pnns_1 = calculate_global_median(train, 'pnns_1', chunk_size)
    print(f"median in train file for pnns_1: {median_pnns_1}")
    median_countries = calculate_global_median(train, 'countries', chunk_size)
    print(f"median in train file for countries: {median_countries}")
    median_nova = calculate_global_median(train, 'nova', chunk_size)
    print(f"median in train file for nova: {median_nova}")
    median_palm_oil = calculate_global_median(train, 'palm_oil', chunk_size)
    print(f"median in train file for palm_oil: {median_palm_oil}")
    median_nutriscore_tags = calculate_global_median(train, 'nutriscore_tags', chunk_size)
    print(f"median in train file for nutriscore_tags: {median_nutriscore_tags}")
    median_additives = calculate_global_median(train, 'additives', chunk_size)
    print(f"median in train file for additives: {median_additives}")

    print("TRAIN")
    train_01 = data_path + file_id + '_train_01.jsonl' 
    print("estimating necessary chunk number train")
    estimated_chunks_train = count_chunks(train, chunk_size)
    print("browse throw train file to process columns")
    browse_file_test_train(estimated_chunks_train, train, train_01, chunk_size, median_countries, median_pnns_1, median_nova, median_palm_oil, median_nutriscore_tags, median_additives)

    print("TEST")
    test = data_path + file_id + '_test.jsonl' 
    test_01 = data_path + file_id + '_test_01.jsonl' 
    print("estimating necessary chunk number test")
    estimated_chunks_test = count_chunks(test, chunk_size)
    print("browse throw test file to process columns")
    browse_file_test_train(estimated_chunks_test, test, test_01, chunk_size, median_countries, median_pnns_1, median_nova, median_palm_oil, median_nutriscore_tags, median_additives)

    print("VALIDATION")
    valid = data_path + file_id + '_valid.jsonl' 
    valid_01 = data_path + file_id + '_valid_01.jsonl' 
    print("estimating necessary chunk number valid")
    estimated_chunks_valid = count_chunks(valid, chunk_size)
    print("browse throw valid file to process columns")
    browse_file_valid(estimated_chunks_valid, valid, valid_01, chunk_size, median_countries, median_pnns_1, median_nova, median_palm_oil, median_nutriscore_tags, median_additives)
    
    #print("deleting file jsonl train 00")
    #delete_file(train)
    #print("deleting file jsonl test 00")
    #delete_file(test)
    #print("deleting file jsonl valid 00")
    #delete_file(valid)

if __name__ == "__main__":
    chunk_size = sys.argv[1]
    file_id = sys.argv[2]
    data_path = sys.argv[3]
    main(chunk_size, file_id, data_path)