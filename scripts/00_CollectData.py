import argparse
import pandas as pd
import requests
from io import BytesIO
import gzip
import os
import shutil

def main(projectPath, fileNbr):
    url = 'https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz'
    compressedFilePath = projectPath + fileNbr + '_OpenFoodFacts.gz'
    csvFilePath = projectPath + fileNbr + '_OpenFoodFacts.csv'

    response = requests.get(url, stream=True)
    with open(compressedFilePath, 'wb') as file:
        shutil.copyfileobj(response.raw, file)

    with gzip.open(compressedFilePath, 'rb') as f_in:
        with open(csvFilePath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    def read_chunks(file_path, chunk_size=100000):
        chunks = []
        try:
            for chunk in pd.read_csv(file_path, sep='\t', chunksize=chunk_size, on_bad_lines='skip'):
                chunks.append(chunk)
        except pd.errors.ParserError as e:
            print(f"erreur lors du parsing: {e}")
        return pd.concat(chunks, ignore_index=True)

    df = read_chunks(csvFilePath)

    print(df.head(10))

    df.to_csv(projectPath + fileNbr + '_OpenFoodFacts.csv', index=False)

    print("fichier csv sauvegardé !")

    if os.path.exists(compressedFilePath):
        os.remove(compressedFilePath)
        print(f"fichier {compressedFilePath} a été supprimé.")
    else:
        print(f"fichier {compressedFilePath} n'existe pas.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="télécharge et traite un fichier Open Food Facts")
    parser.add_argument('projectPath', type=str, help='le chemin du projet où les fichiers seront sauvegardés')
    parser.add_argument('fileNbr', type=str, help="numéro d'identification des csv à générer")
    
    args = parser.parse_args()
    main(args.projectPath, args.fileNbr)
