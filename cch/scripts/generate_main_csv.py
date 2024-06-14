import pandas as pd
import requests
from io import BytesIO
import gzip

# panneau de configuration
project_path = "C:\\Users\\charl\\Documents\\workspace\\green_ia\\cch\\"
file_nbr = '02' # numéro d'identification des csv à générer 

# récupére données open food facts à jour en ligne 
url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
response = requests.get(url)

if response.status_code == 200:
    with gzip.GzipFile(fileobj=BytesIO(response.content), mode='rb') as file:
        df = pd.read_csv(file, sep='\t', encoding='utf-8')
        pd.set_option('display.max_columns', None)
        print(df)
        
        # sauvegarde en local dans un csv
        df.to_csv(project_path + f'data_global\\openfoodfacts_{file_nbr}.csv', index=False)  # Vous pouvez spécifier l'index comme False si vous ne voulez pas le sauvegarder
        print("csv sauvegardé en local")
else:
    print(f"error, {response.status_code}.")