import requests
import gzip
import pandas as pd
from io import BytesIO
import os
import shutil
from requests.exceptions import RequestException
import warnings
from datetime import datetime

pd.set_option('display.max_rows', 50)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)





url = "https://static.openfoodfacts.org/data/openfoodfacts-products.jsonl.gz"
fileNbr = '01'
projectPath = "/home/carolus/Documents/school/green_ia/" 
jsonGz = projectPath + "data/" + fileNbr + "_openfoodfacts" + ".jsonl.gz"
linesPerFile = 10000 # nombre de ligne pour chaque petit csv
csvPath = projectPath + "data/" + fileNbr + "_openfoodfacts_00/" 

colToSave = ['allergens_from_ingredients',
            'nutriscore_tags',
            'labels_old',
            'categories_old',
            'pnns_groups_1',
            'ecoscore_data',
            'brand_owner_imported',
            'ingredients_tags',
            'packaging',
            'ingredients_hierarchy',
            'product_name',
            'food_groups_tags',
            'ecoscore_tags',
            'nova_group',
            'ingredients_from_or_that_may_be_from_palm_oil_n',
            'categories_tags',
            'brand_owner',
            'nutrient_levels_tags',
            'allergens_tags',
            'ecoscore_extended_data',
            'categories',
            'nutriments',
            'nutriscore_2021_tags',
            'additives_old_n',
            'ecoscore_score',
            'labels_tags',
            'countries']

# récupérer la date du jour 
currentDateTime = datetime.now()
formattedDate = currentDateTime.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
dateCode = currentDateTime.strftime('%d%m%Y%H%M%S') + f"{currentDateTime.microsecond // 1000:03d}"







def addLogs(logData):
    print(logData)
    with open(f"{projectPath}logs/{dateCode}_logs.txt", "a") as logFile:
        logFile.write(f'{logData}\n')

addLogs("parameters used by user:")
addLogs(f"start date: {formattedDate}")
addLogs(f"url: {url} \nfileNbr: {fileNbr} \nprojectPath: {projectPath} \njsonGz: {jsonGz} \nlinesPerFile: {linesPerFile} \ncsvPath: {csvPath} \ncolToSave: {colToSave}")

# fonction pour reprendre le téléchargement
def downloadFile(url, jsonGz):
    addLogs("start downloading file from Open Food Facts")
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
                addLogs(f"downloaded: {jsonGz}")
                break  # sortir boucle une fois téléchargement terminé

            else:
                addLogs(f"ERROR while downloading: {response.status_code}")
                break  # sortir boucle si erreur statut

        except RequestException as e:
            addLogs(f"warning, continue downloading: {e}")

# décompresser du fichier jsonl
def unGzFile(jsonGz, fileNbr, projectPath):
    addLogs("start unzziping jsonl compressed")
    jsonl = projectPath + "data/" + fileNbr + '_openfoodfacts.jsonl'
    with gzip.open(jsonGz, 'rb') as f_in:
        with open(jsonl, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    addLogs(f'unzziping completed: {jsonl}')
    return jsonl

# conversion en fichier csv
def convertToCsv(jsonl, fileNbr, projectPath):
    addLogs("converting jsonl file to csv file")
    heavyCsv = projectPath + "data/" + fileNbr + '_openfoodfacts_00.csv'
    chunksize = 10000  
    chunkIter = pd.read_json(jsonl, lines=True, chunksize=chunksize)

    for i, chunk in enumerate(chunkIter):
        if i == 0:
            chunk.to_csv(heavyCsv, index=False, escapechar='\\')
        else:
            chunk.to_csv(heavyCsv, mode='a', header=False, index=False, escapechar='\\')

    addLogs(f"convert jsonl to heavy csv terminated: {heavyCsv}")
    return heavyCsv

def deleteFile(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)
        addLogs(f"file deleted: {filePath}")
    else:
        addLogs(f"ERROR, does not exists: {filePath}")

def createFolder(folderPath):
    try:
        os.makedirs(folderPath, exist_ok=True)
        addLogs(f"new folder: {folderPath}")
    except OSError as e:
        addLogs(f"ERROR while creating folder: {e}")

def splitCsv(csvFile, linesPerFile, csvPath):
    addLogs("spliting heavy csv to several small csv")
    try:
        chunkSize = linesPerFile
        chunks = pd.read_csv(csvFile, chunksize=chunkSize, on_bad_lines='skip')
        
        allCol = set()
        for chunk in chunks:
            allCol.update(chunk.columns)
        
        chunks = pd.read_csv(csvFile, chunksize=chunkSize, on_bad_lines='skip')
        
        for i, chunk in enumerate(chunks):
            for col in allCol:
                if col not in chunk.columns:
                    chunk[col] = None
            chunk = chunk[list(allCol)]
            
            outputFile = f"{csvPath}{i+1}_openfoodfacts_00.csv"
            chunk.to_csv(outputFile, index=False)
            addLogs(f"small csv generated: {outputFile}")
            
    except Exception as e:
        addLogs(f"ERROR while spliting heavy csv: {e}")

# compter les fichiers csv dans le dossier traité 
def countCsv(directory):
    addLogs("counting small csv file in specific folder")
    csvNbr = 0
    for csvFile in os.listdir(directory):
        if csvFile.endswith('.csv'):
            csvNbr += 1
    return csvNbr

# lister noms de colonnes et les sauvegarder dans un fichier texte
def findAndSaveCol(df, csvPath):
    addLogs("listing columns names and save it in text file")
    colName = df.columns.tolist()

    colTextFile = csvPath + "colSaver.txt"
    with open(colTextFile, 'w') as file:
        for name in colName:
            file.write(name + " | ")




# main algo
downloadFile(url, jsonGz)
jsonl = unGzFile(jsonGz, fileNbr, projectPath)
deleteFile(jsonGz)
heavyCsv = convertToCsv(jsonl, fileNbr, projectPath)
deleteFile(jsonl)
createFolder(csvPath) 
splitCsv(heavyCsv, linesPerFile, csvPath)

# compte le nombre de csv dans dossier 00
csvNbr = countCsv(csvPath)
addLogs(f'count small csv step 00: {csvNbr}')

csvIterator = 1
while csvIterator < csvNbr:
    # initialise df
    df_00, df_01 = pd.DataFrame(), pd.DataFrame()

    currentCsv = f"{csvPath}{csvIterator}_openfoodfacts_00.csv"
    addLogs(f"current csv: {currentCsv}")

    # traitement ici 
    df_00 = pd.read_csv(currentCsv)

    # lister toutes les colonnes du premier csv et les sauvegarder dans un fichier texte
    if csvIterator == 1:
        findAndSaveCol(df_00, csvPath)

    # garde dans df uniquement les colonnes utiles
    df_01 = df_00[colToSave]

    # générer un fichier csv bis avec les colonnes utiles uniquement (setp 01)
    df_01.to_csv(f"{csvPath}{csvIterator}_openfoodfacts_01.csv", index=False)
    addLogs(f"small csv step 01 generated: {currentCsv}")

    # supprimer le fichier csv initial 
    deleteFile(currentCsv)

    csvIterator+=1

# compte le nombre de csv dans dossier 01
csvNbr = countCsv(csvPath)
addLogs(f'count small csv step 01: {csvNbr}')

currentDateTime = datetime.now()
formattedDate = currentDateTime.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
addLogs(f"end date: {formattedDate}")
