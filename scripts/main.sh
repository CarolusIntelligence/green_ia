#!/bin/bash
clear

start_time=$(date "+%Y-%m-%d_%H:%M:%S")
start_seconds=$(date +%s)
log_file="../logs/${start_time}_logs.txt"
echo "begin: $start_time" > "$log_file"

download_url=$(jq -r '.download_url' config.json)
file_id=$(jq -r '.file_id' config.json)
data_path=$(jq -r '.data_path' config.json)
scripts_path=$(jq -r '.scripts_path' config.json)
logs_path=$(jq -r '.logs_path' config.json)
chunk_size=$(jq -r '.chunk_size' config.json)

data_path="${data_path}${file_id}_data/"

{
  #echo "exec 00_collect_data.py"
  #python 00_collect_data.py "$download_url" "$file_id" "$data_path" "$chunk_size"
  #echo "exec 01_keep_usefull_columns.py"
  #python 01_keep_usefull_columns.py "$chunk_size" "$file_id" "$data_path"
  #echo "exec 02_columns_preprocessing.py"
  #python 02_columns_preprocessing.py "$chunk_size" "$file_id" "$data_path" "$scripts_path"
  #echo "exec 03_nlp_processing.py"
  #python 03_nlp_processing.py "$file_id" "$data_path" "$chunk_size"
  echo "exec 04_split_dataset.py"
  python 04_split_dataset.py "$chunk_size" "$file_id" "$data_path"
} #>> "$log_file" 2>&1

end_time=$(date "+%Y-%m-%d %H:%M:%S")
end_seconds=$(date +%s)
duration=$((end_seconds - start_seconds))
{
  echo "end: $end_time"
  echo "total execution time: $((duration / 60)) minutes $((duration % 60)) seconds"
} #>> "$log_file"
