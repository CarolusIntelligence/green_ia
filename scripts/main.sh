#!/bin/bash

# lecture fichier de configuration 
download_url=$(jq -r '.download_url' config.json)
file_id=$(jq -r '.file_id' config.json)
project_path=$(jq -r '.project_path' config.json)
chunk_size=$(jq -r '.chunk_size' config.json)

# execution des scripts python 
python ./00_collect_data.py "$download_url" "$file_id" "$project_path"
#python ./01_preprocessing.py "$chunk_size" "$file_id" "$project_path"
#python ./03_data_analysis.py "$chunk_size" "$file_id" "$project_path"
