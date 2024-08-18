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


download_url = "https://static.openfoodfacts.org/data/openfoodfacts-products.jsonl.gz"
file_id = '02'
project_path = "/home/carolus/Documents/school/green_ia/" 
jsonl_gz = project_path + "data/" + file_id + "_openfoodfacts_00" + ".jsonl.gz"


# récupérer la date du jour 
current_date_time = datetime.now()
date_format = "%d/%m/%Y %H:%M:%S.%f"
start_date = current_date_time.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
date_code = current_date_time.strftime('%d%m%Y%H%M%S') + f"{current_date_time.microsecond // 1000:03d}"


def add_logs(logData):
    print(logData)
    with open(f"{project_path}logs/00_collect_data_{date_code}_logs.txt", "a") as logFile:
        logFile.write(f'{logData}\n')


# fonction pour reprendre le téléchargement
def download_file(download_url, jsonl_gz):
    add_logs("start downloading file from Open Food Facts data base")
    while True:
        try:
            # vérifier si fichier existe déjà et obtenir sa taille
            file_size = 0
            if os.path.exists(jsonl_gz):
                file_size = os.path.getsize(jsonl_gz)

            headers = {"range": f"bytes={file_size}-"}
            response = requests.get(download_url, headers=headers, stream=True)

            if response.status_code in [200, 206]:
                mode = 'ab' if file_size else 'wb'
                with open(jsonl_gz, mode) as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                add_logs(f"downloaded: {jsonl_gz}")
                break  # sortir boucle une fois téléchargement terminé

            else:
                add_logs(f"ERROR while downloading: {response.status_code}")
                break  # sortir boucle si erreur statut

        except RequestException as e:
            add_logs(f"warning, continue downloading: {e}")


# décompresser du fichier jsonl
def un_gz_file(jsonl_gz, file_id, project_path):
    add_logs("start unzziping jsonl compressed")
    jsonl = project_path + "data/" + file_id + '_openfoodfacts_00.jsonl'
    with gzip.open(jsonl_gz, 'rb') as f_in:
        with open(jsonl, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    add_logs(f'unzziping completed: {jsonl}')
    return jsonl


def get_time():
    current_date = datetime.now()
    current_date = current_date.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
    return current_date


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        add_logs(f"file deleted: {file_path}")
    else:
        add_logs(f"ERROR, does not exists: {file_path}")


add_logs(f"start time downloading: {get_time()}")
add_logs("00_collect_data logs:")
add_logs(f"download_url: {download_url} \nfile_id: {file_id} \nproject_path: {project_path} \njsonl_gz: {jsonl_gz}")


# main algo
download_file(download_url, jsonl_gz)
jsonl = un_gz_file(jsonl_gz, file_id, project_path)
delete_file(jsonl_gz)


# récupérer la date du jour 
current_date_time = datetime.now()
end_date = current_date_time.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
add_logs(f"end date: {end_date}")

# afficher temps total execution script 
start_date = datetime.strptime(start_date, date_format)
end_date = datetime.strptime(end_date, date_format)
time_difference = end_date - start_date
time_difference_minutes = time_difference.total_seconds() / 60
add_logs(f"execution script time: {time_difference_minutes:.2f} minutes")