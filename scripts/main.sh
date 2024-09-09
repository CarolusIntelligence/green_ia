#!/bin/bash
clear

start_time=$(date "+%Y-%m-%d_%H-%M-%S")
start_seconds=$(date +%s)
log_file="../logs/${start_time}_logs.txt"
echo "begin: $start_time" # > "$log_file" -------------------------------------------------------------------------------> ICI

download_url=$(jq -r '.download_url' config.json)
file_id=$(jq -r '.file_id' config.json)
data_path=$(jq -r '.data_path' config.json)
scripts_path=$(jq -r '.scripts_path' config.json)
logs_path=$(jq -r '.logs_path' config.json)
chunk_size=$(jq -r '.chunk_size' config.json)
MAX_SEQ_LEN=$(jq -r '.MAX_SEQ_LEN' config.json)
batch_size=$(jq -r '.batch_size' config.json)
embed_dim=$(jq -r '.embed_dim' config.json)
hidden_dim=$(jq -r '.hidden_dim' config.json)
lr=$(jq -r '.lr' config.json)
patience=$(jq -r '.patience' config.json)
best_model_path=$(jq -r '.best_model_path' config.json)

data_path="${data_path}${file_id}_data/"
best_model_path="${best_model_path}${file_id}_${MAX_SEQ_LEN}_${batch_size}_${embed_dim}_${hidden_dim}_${lr}.ci"

{
  # CREATION DATASET 
  #echo "exec 00_collect_data.py"
  #python 00_collect_data.py "$download_url" "$file_id" "$data_path" "$chunk_size" 
  #echo "exec 01_keep_usefull_columns.py"
  #python 01_keep_usefull_columns.py "$chunk_size" "$file_id" "$data_path" 
  #echo "exec 02_columns_preprocessing.py"
  #python 02_columns_preprocessing.py "$chunk_size" "$file_id" "$data_path" "$scripts_path"
  #echo "exec 03_split_dataset.py"
  #python 03_split_dataset.py "$chunk_size" "$file_id" "$data_path"
  #echo "exec 04_norm_impuNaN.py"
  #python 04_norm_impuNaN.py "$chunk_size" "$file_id" "$data_path"

  # ENTRAINEMENT MODELE IA 
  #echo "exec 05_pytorch_pred_score.py"
  #python 05_pytorch_pred_score.py "$chunk_size" "$file_id" "$data_path" "$MAX_SEQ_LEN" "$batch_size" "$embed_dim" "$hidden_dim" "$lr" "$patience" "$best_model_path"
  #echo "exec 06_pytorch_validation_model.py"
  #python 06_pytorch_validation_model.py "$file_id" "$data_path"

} # >> "$log_file" 2>&1 -------------------------------------------------------------------------------> ICI

end_time=$(date "+%Y-%m-%d %H:%M:%S")
end_seconds=$(date +%s)
duration=$((end_seconds - start_seconds))
{
  echo "end: $end_time"
  echo "total execution time: $((duration / 60)) minutes $((duration % 60)) seconds"
}  # >> "$log_file" -------------------------------------------------------------------------------> ICI
