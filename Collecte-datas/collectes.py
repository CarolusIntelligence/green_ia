import os
import pandas as pd
from process_data import process_data
import json
# Dossier où sont stockés les fichiers JSON
input_folder = 'products_json/'

# Liste pour stocker les DataFrames de chaque fichier JSON
dataframes = []

# Itérer sur les fichiers JSON dans le dossier
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)

        # Lire le fichier JSON dans un DataFrame
        df = pd.read_json(file_path, orient='index')
        dataframes.append(df)

# Concaténer tous les DataFrames en un seul
result_df = pd.concat(dataframes, axis=1).T

# Réinitialiser l'indice du DataFrame
result_df.reset_index(drop=True, inplace=True)

# Sauvegarder le DataFrame résultant dans un seul fichier JSON
result_json_path = 'export_json.json'
result_df.to_json(result_json_path, orient='records')  # Utiliser orient='records'
flattened_data = process_data(result_df)
with open('export_json.json', 'w') as f:
    json.dump(flattened_data, f)
# Afficher le chemin du fichier JSON créé pour l'ensemble des produits
#print(f"Le fichier JSON de l'ensemble des produits a été créé avec succès : {result_json_path}")
