#!/bin/bash

clear

# lecture fichier de configuration 
download_url=$(jq -r '.download_url' config.json)
file_id=$(jq -r '.file_id' config.json)
data_path=$(jq -r '.data_path' config.json)
scripts_path=$(jq -r '.scripts_path' config.json)
logs_path=$(jq -r '.logs_path' config.json)
chunk_size=$(jq -r '.chunk_size' config.json)


data_path="${data_path}${file_id}_data/"


# execution des scripts python 
#python ./00_collect_data.py "$download_url" "$file_id" "$data_path" "$chunk_size"
#python ./01_keep_usefull_columns.py "$chunk_size" "$file_id" "$data_path"
python ./02_columns_preprocessing.py "$chunk_size" "$file_id" "$data_path"