import requests
import json

# Liste de codes de produits (remplacez-les par les codes réels que vous souhaitez interroger)
product_codes = [
    "7622210449283",
    "1234567890123",
    "9876543210987",
    # Ajoutez ici les codes des 27 produits restants
    "1111111111111",
    "2222222222222",
    "3333333333333",
    "4444444444444",
    "5555555555555",
    "6666666666666",
    "7777777777777",
    "8888888888888",
    "9999999999999",
    "1010101010101",
    "2020202020202",
    "3030303030303",
    "4040404040404",
    "5050505050505",
    "6060606060606",
    "7070707070707",
    "8080808080808",
    "9090909090909",
    "1212121212121",
    "1313131313131",
    "1414141414141",
    "1515151515151",
    "1616161616161",
]

# Dossier pour stocker les fichiers JSON
output_folder = 'products_json/'

# Créer le dossier s'il n'existe pas
import os
os.makedirs(output_folder, exist_ok=True)

# Itérer sur les codes de produits et effectuer une requête pour chaque produit
for i, product_code in enumerate(product_codes, 1):
    # URL de l'API Open Food Facts pour un produit spécifique
    api_url = f"https://world.openfoodfacts.org/api/v0/product/{product_code}.json"

    # Effectuer une requête GET à l'API pour obtenir les données d'un produit
    response = requests.get(api_url)

    # Vérifier si la requête a réussi (code de statut 200)
    if response.status_code == 200:
        # Récupérer les données JSON de la réponse
        product_data = response.json()

        # Sauvegarder les informations du produit dans un fichier JSON spécifique
        json_filename = f'{output_folder}product_info_{i}.json'
        with open(json_filename, 'w') as json_file:
            json.dump(product_data, json_file, indent=2)
    else:
        print(f"La requête a échoué avec le code de statut {response.status_code} pour le produit {product_code}")
