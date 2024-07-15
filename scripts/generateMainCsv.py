import argparse
import pandas as pd
import requests
from io import BytesIO
import gzip
import os
import shutil

def main(project_path, file_nbr):
    url = 'https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz'
    compressed_file_path = os.path.join(project_path, f'openfoodfacts_{file_nbr}.gz')
    csv_file_path = os.path.join(project_path, f'openfoodfacts_{file_nbr}.csv')

    response = requests.get(url, stream=True)
    with open(compressed_file_path, 'wb') as file:
        shutil.copyfileobj(response.raw, file)

    with gzip.open(compressed_file_path, 'rb') as f_in:
        with open(csv_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    def read_chunks(file_path, chunk_size=100000):
        chunks = []
        try:
            for chunk in pd.read_csv(file_path, sep='\t', chunksize=chunk_size, on_bad_lines='skip'):
                chunks.append(chunk)
        except pd.errors.ParserError as e:
            print(f"erreur lors du parsing: {e}")
        return pd.concat(chunks, ignore_index=True)

    df = read_chunks(csv_file_path)

    print(df.head(10))

    df.to_csv(os.path.join(project_path, f'openfoodfacts_{file_nbr}.csv'), index=False)

    print("fichier csv sauvegardé !")

    if os.path.exists(compressed_file_path):
        os.remove(compressed_file_path)
        print(f"fichier {compressed_file_path} a été supprimé.")
    else:
        print(f"fichier {compressed_file_path} n'existe pas.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="télécharge et traite un fichier Open Food Facts")
    parser.add_argument('project_path', type=str, help='le chemin du projet où les fichiers seront sauvegardés')
    parser.add_argument('file_nbr', type=str, help="numéro d'identification des csv à générer")
    
    args = parser.parse_args()
    main(args.project_path, args.file_nbr)
