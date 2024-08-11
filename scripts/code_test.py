import pandas as pd
import random

# Chemin vers le fichier filtré
output_file_path = '/home/carolus/Documents/school/green_ia/data/filtered_openfoodfacts.jsonl'
mini_file_path = '/home/carolus/Documents/school/green_ia/data/mini_openfoodfacts.jsonl'

# Taille du chunk (nombre de lignes à lire à la fois)
chunk_size = 10000

# Liste pour stocker les échantillons aléatoires
sampled_rows = []

# Lecture du fichier en chunks
for chunk in pd.read_json(output_file_path, lines=True, chunksize=chunk_size):
    # Échantillonner quelques lignes aléatoires dans chaque chunk
    sampled_chunk = chunk.sample(n=min(1000, len(chunk)), random_state=random.randint(1, 10000))
    sampled_rows.append(sampled_chunk)

# Concaténer tous les échantillons
sampled_df = pd.concat(sampled_rows)

# Si le total des lignes échantillonnées est supérieur à 50, échantillonner à nouveau pour réduire à 50
if len(sampled_df) > 1000:
    sampled_df = sampled_df.sample(n=1000, random_state=42)

# Sauvegarder les 50 lignes échantillonnées dans un nouveau fichier JSONL
sampled_df.to_json(mini_file_path, orient='records', lines=True)

print(f"Un mini fichier JSONL contenant 1000 lignes a été créé : {mini_file_path}")
