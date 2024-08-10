import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sea
import sklearn
import scipy as sc
import nltk as nltk
import statsmodels as statsmodels
import os
import warnings
import csv
from datetime import datetime 

pd.set_option('display.max_rows', 50)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import dask
print(dask.__version__)






import jsonlines

# Chemin vers le fichier JSONL
file_path = '/home/carolus/Documents/school/green_ia/data/04_openfoodfacts.jsonl'

# Liste pour stocker les premiers éléments
data = []

# Lire les 10 premiers éléments du fichier JSONL
with jsonlines.open(file_path) as reader:
    for i, obj in enumerate(reader):
        if i < 10:  # Lire seulement les 10 premiers éléments
            data.append(obj)
        else:
            break

# Créer un DataFrame à partir des données
df = pd.DataFrame(data)

# Afficher les premières lignes du DataFrame pour vérification
df.head()



