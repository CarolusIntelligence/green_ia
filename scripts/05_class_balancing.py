import pandas as pd
import json
import sys


def load_jsonl_data_in_batches(filepath, batch_size):
    batch = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            batch.append(json.loads(line))
            if (i + 1) % batch_size == 0:
                yield pd.DataFrame(batch)
                batch = []
    if batch:
        yield pd.DataFrame(batch)

def balance_classes_in_batch(df, target_column):
    class_counts = df[target_column].value_counts()
    min_class_size = class_counts.min()
    
    balanced_df = df.groupby(target_column).apply(lambda x: x.sample(min_class_size)).reset_index(drop=True)
    
    return balanced_df

def save_jsonl_data(df, filepath):
    with open(filepath, 'a') as f:
        for record in df.to_dict(orient='records'):
            json.dump(record, f)
            f.write('\n')

def process_file_in_batches(input_path, output_path, batch_size, target_column):
    with open(output_path, 'w'):
        pass
    all_balanced_df = []
    for batch_df in load_jsonl_data_in_batches(input_path, batch_size):
        balanced_df = balance_classes_in_batch(batch_df, target_column)
        all_balanced_df.append(balanced_df)
        save_jsonl_data(balanced_df, output_path)
    return pd.concat(all_balanced_df, ignore_index=True)

def display_class_counts(df, filename, target_column):
    print(f"\nline number in: {filename} = ")
    print(df[target_column].value_counts())

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"file deleted: {file_path}")
    else:
        print(f"ERROR, does not exists: {file_path}")



###############################################################################
# MAIN ########################################################################
###############################################################################
def main(chunk_size, file_id, data_path):
    batch_size = int(chunk_size)
    train_data_path = data_path + file_id + "_train_01.jsonl"
    test_data_path = data_path + file_id + "_test_01.jsonl"
    valid_data_path = data_path + file_id + "_valid_01.jsonl"
    target_column = 'ecoscore_tags'

    train_balanced_df = process_file_in_batches(train_data_path, data_path + file_id + "_train_02.jsonl", batch_size, target_column)
    test_balanced_df = process_file_in_batches(test_data_path, data_path + file_id + "_test_02.jsonl", batch_size, target_column)
    valid_balanced_df = process_file_in_batches(valid_data_path, data_path + file_id + "_valid_02.jsonl", batch_size, target_column)

    display_class_counts(train_balanced_df, "_train_02.jsonl", target_column)
    display_class_counts(test_balanced_df, "_test_02.jsonl", target_column)
    display_class_counts(valid_balanced_df, "_valid_02.jsonl", target_column)   
    
    print(f"deleting {train_data_path}")
    delete_file(train_data_path)
    print(f"deleting {test_data_path}")
    delete_file(test_data_path)
    print(f"deleting {valid_data_path}")
    delete_file(valid_data_path)

if __name__ == "__main__":
    chunk_size = sys.argv[1]
    file_id = sys.argv[2]
    data_path = sys.argv[3]
    main(chunk_size, file_id, data_path)
