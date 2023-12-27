import requests
import csv
import json

# Déclarer le code du produit que vous souhaitez interroger
product_code = "7622210449283"  # Remplacez ceci par le code de produit que vous souhaitez récupérer

# URL de l'API Open Food Facts
api_url = f"https://world.openfoodfacts.org/api/v0/product/{product_code}.json"

# Effectuer une requête GET à l'API
response = requests.get(api_url)

# Vérifier si la requête a réussi (code de statut 200)
if response.status_code == 200:
    # Récupérer les données JSON de la réponse
    product_data = response.json()

    # Sauvegarder les données dans un fichier JSON
    with open('export.json', 'w') as file:
        json.dump(product_data, file)
else:
    print(f"La requête a échoué avec le code de statut {response.status_code}")






