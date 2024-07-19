import requests
import gzip
import pandas as pd
from io import BytesIO
import os
import shutil
from requests.exceptions import RequestException


url = "https://static.openfoodfacts.org/data/openfoodfacts-products.jsonl.gz"
fileNbr = '01'
projectPath = "/home/carolus/Documents/school/green_ia/" 
jsonGz = projectPath + "data/" + fileNbr + "_openfoodfacts" + ".jsonl.gz"


# fonction pour reprendre le téléchargement
def downloadFile(url, jsonGz):
    while True:
        try:
            # vérifier si fichier existe déjà et obtenir sa taille
            fileSize = 0
            if os.path.exists(jsonGz):
                fileSize = os.path.getsize(jsonGz)

            headers = {"range": f"bytes={fileSize}-"}
            response = requests.get(url, headers=headers, stream=True)

            if response.status_code in [200, 206]:
                mode = 'ab' if fileSize else 'wb'
                with open(jsonGz, mode) as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                print(f"fichier téléchargé et sauvegardé ici: {jsonGz}")
                break  # sortir boucle une fois téléchargement terminé

            else:
                print(f"erreur: {response.status_code}")
                break  # sortir boucle si erreur statut

        except RequestException as e:
            print(f"pause, reprise du téléchargement : {e}")

# décompresser du fichier jsonl
def unGzFile(jsonGz, fileNbr, projectPath):
    jsonl = projectPath + "data/" + fileNbr + '_openfoodfacts.jsonl'
    with gzip.open(jsonGz, 'rb') as f_in:
        with open(jsonl, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f'fichier {jsonl} décompressé avec succès')
    return jsonl

# conversion en fichier csv
def convertToCsv(jsonl, fileNbr, projectPath):
    csv = projectPath + "data/" + fileNbr + '_openfoodfacts_00.csv'
    chunksize = 10000  
    chunkIter = pd.read_json(jsonl, lines=True, chunksize=chunksize)

    for i, chunk in enumerate(chunkIter):
        if i == 0:
            chunk.to_csv(csv, index=False, escapechar='\\')
        else:
            chunk.to_csv(csv, mode='a', header=False, index=False, escapechar='\\')

    print("conversion vers csv terminée")

def deleteFile(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)
        print(f"fichier {filePath} supprimé")
    else:
        print(f"erreur, fichier {filePath} n'existe pas")


downloadFile(url, jsonGz)
jsonl = unGzFile(jsonGz, fileNbr, projectPath)
deleteFile(jsonGz)
convertToCsv(jsonl, fileNbr, projectPath)
deleteFile(jsonl)