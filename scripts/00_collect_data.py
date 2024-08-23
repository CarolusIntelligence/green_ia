import requests
import gzip
import os
import shutil
from requests.exceptions import RequestException
import sys


# create folder 
def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"folder: {folder} successfully created")
    else:
        print(f"ERROR, folder: {folder} already exists")
        sys.exit()

# fonction pour reprendre le téléchargement
def download_file(download_url, jsonl_gz, chunk_size):
        while True:
            try:
                file_size = 0
                if os.path.exists(jsonl_gz):
                    file_size = os.path.getsize(jsonl_gz)
                headers = {"range": f"bytes={file_size}-"}
                response = requests.get(download_url,
                                        headers=headers, 
                                        stream=True)
                if response.status_code in [200, 206]:
                    mode = 'ab' if file_size else 'wb'
                    with open(jsonl_gz, mode) as file:
                        for chunk in response.iter_content(chunk_size):
                            if chunk:
                                file.write(chunk)
                    print(f"downloaded: {jsonl_gz}")
                    break  # sortir boucle une fois téléchargement terminé
                else:
                    print(f"ERROR while downloading: {response.status_code}")
                    break  # sortir boucle si erreur statut
            except RequestException as e:
                print(f"warning, continue downloading: {e}")

# décompresser du fichier jsonl
def un_gz_file(file_id, data_path, jsonl_gz, jsonl):
    with gzip.open(jsonl_gz, 'rb') as f_in:
        with open(jsonl, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f'unzziping completed: {jsonl}')
    return jsonl_gz

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"file deleted: {file_path}")
    else:
        print(f"ERROR, does not exists: {file_path}")


###############################################################################
# MAIN ########################################################################
###############################################################################
def main(download_url, file_id, data_path, chunk_size):
    chunk_size = int(chunk_size)
    jsonl_gz = data_path + file_id + "_openfoodfacts_00" + ".jsonl.gz"
    jsonl = data_path + file_id + '_openfoodfacts_01.jsonl'

    print("create folder")
    create_folder(data_path)
    print("start downloading jsonl file from open food facts data-base")
    download_file(download_url, jsonl_gz, chunk_size)
    print("uncompress jsonl file")
    un_gz_file(file_id, data_path, jsonl_gz, jsonl)
    print("delete jsonl file compressed")
    delete_file(jsonl_gz)

if __name__ == "__main__":
    download_url = sys.argv[1]
    file_id = sys.argv[2]
    data_path = sys.argv[3] 
    chunk_size = sys.argv[4]   
    main(download_url, file_id, data_path, chunk_size)

