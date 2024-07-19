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
linesPerFile = 10000 # nombre de ligne pour chaque petit csv
csvOutput = projectPath + "data/" + fileNbr + "_openfoodfacts_00/" # dossier des minis csv

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
    heavyCsv = projectPath + "data/" + fileNbr + '_openfoodfacts_00.csv'
    chunksize = 10000  
    chunkIter = pd.read_json(jsonl, lines=True, chunksize=chunksize)

    for i, chunk in enumerate(chunkIter):
        if i == 0:
            chunk.to_csv(heavyCsv, index=False, escapechar='\\')
        else:
            chunk.to_csv(heavyCsv, mode='a', header=False, index=False, escapechar='\\')

    print("conversion vers csv terminée")
    return heavyCsv

def deleteFile(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)
        print(f"fichier {filePath} supprimé")
    else:
        print(f"erreur, fichier {filePath} n'existe pas")

def createFolder(folderPath):
    try:
        os.makedirs(folderPath, exist_ok=True)
        print(f"dossier créé avec succès: {folderPath}")
    except OSError as e:
        print(f"erreur création du dossier: {e}")

def splitCsv(csvFile, linesPerFile, csvOutput):
    try:
        chunkSize = linesPerFile
        chunks = pd.read_csv(csvFile, chunksize=chunkSize, on_bad_lines='skip')
        
        all_columns = set()
        for chunk in chunks:
            all_columns.update(chunk.columns)
        
        chunks = pd.read_csv(csvFile, chunksize=chunkSize, on_bad_lines='skip')
        
        for i, chunk in enumerate(chunks):
            for col in all_columns:
                if col not in chunk.columns:
                    chunk[col] = None
            chunk = chunk[list(all_columns)]
            
            outputFile = f"{csvOutput}{i+1}_openfoodfacts_00.csv"
            chunk.to_csv(outputFile, index=False)
            print(f"fichier {outputFile} sauvegardé avec {len(chunk)} lignes")
            
    except Exception as e:
        print(f"warning")



downloadFile(url, jsonGz)
jsonl = unGzFile(jsonGz, fileNbr, projectPath)
deleteFile(jsonGz)
heavyCsv = convertToCsv(jsonl, fileNbr, projectPath)
deleteFile(jsonl)
createFolder(csvOutput) 
splitCsv(heavyCsv, linesPerFile, csvOutput)