import os
import json
import sys


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"file jsonl 01 deleted: {file_path}")
    else:
        print(f"ERROR, does not exists: {file_path}")

# génération jsonl filtré
def jsonl_filtered_creator(jsonl_01, columns_to_keep, jsonl_02, chunk_size):
    with open(jsonl_01, 
            'r', 
            encoding='utf-8') as infile, open(jsonl_02, 'w', 
                                                encoding='utf-8') as outfile:
        buffer = []
        for i, line in enumerate(infile):
            record = json.loads(line.strip())        
            filtered_record = {key: record.get(key) for key in columns_to_keep}        
            buffer.append(json.dumps(filtered_record) + '\n')
            if len(buffer) >= chunk_size:
                outfile.writelines(buffer)
                buffer = []
        if buffer:
            outfile.writelines(buffer)
    print(f"jsonl 02 generated: {jsonl_02}")


###############################################################################
# MAIN ########################################################################
###############################################################################
def main(chunk_size, file_id, data_path):
    chunk_size = int(chunk_size)
    jsonl_01 = data_path + file_id + "_openfoodfacts_01" + ".jsonl" 
    jsonl_02 = data_path + file_id + '_openfoodfacts_02.jsonl' 
    columns_to_keep = ['pnns_groups_1',
                'ingredients_tags',
                'packaging',
                'product_name',
                'ecoscore_tags',
                'categories_tags',
                'ecoscore_score',
                'labels_tags',
                'code',
                'countries']

    print("generating jsonl 02 with only usefull columns")
    jsonl_filtered_creator(jsonl_01, columns_to_keep, jsonl_02, chunk_size)
    print("deleting file jsonl 01")
    delete_file(jsonl_01)

if __name__ == "__main__":
    chunk_size = sys.argv[1]
    file_id = sys.argv[2]
    data_path = sys.argv[3]
    main(chunk_size, file_id, data_path)