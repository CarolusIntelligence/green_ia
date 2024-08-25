import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, concatenate, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Dense, Embedding, concatenate, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.preprocessing import LabelEncoder
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, concatenate, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.preprocessing import LabelEncoder

# Définir les fichiers de données
train_file = '/home/carolus/Documents/school/green_ia/data/01_data/01_train.jsonl'
valid_file = '/home/carolus/Documents/school/green_ia/data/01_data/01_valid.jsonl'
test_file = '/home/carolus/Documents/school/green_ia/data/01_data/01_test.jsonl'

# Fonction pour charger les données par lots depuis un fichier JSONL
def load_jsonl(file):
    with open(file, 'r') as f:
        for line in f:
            yield json.loads(line)

# Encodage des colonnes catégorielles
def encode_column(values):
    le = LabelEncoder()
    # Ajouter une valeur par défaut 'Unknown' pour les données manquantes
    values = list(values) + ['Unknown']
    le.fit(values)
    return le

# Créer un encodeur pour chaque colonne catégorielle
categorical_columns = ['groups', 'packaging', 'countries', 'categories']
label_encoders = {}

for col in categorical_columns:
    # Charger toutes les valeurs uniques pour cet encodeur
    values = set()
    for entry in load_jsonl(train_file):
        if entry[col] is not None:
            values.add(entry[col])
    label_encoders[col] = encode_column(values)

# Préparation des TextVectorizers pour les colonnes textuelles
def vectorize_text(column, max_tokens=20000, output_sequence_length=20):
    vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_sequence_length)
    
    # Créer un Dataset TensorFlow avec uniquement les textes de la colonne
    def text_generator():
        for entry in load_jsonl(train_file):
            # Remplacer None par une chaîne vide ""
            yield entry[column] if entry[column] is not None else ""

    text_ds = tf.data.Dataset.from_generator(
        text_generator,
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
    ).batch(128)  # Batch pour l'adaptation

    vectorizer.adapt(text_ds)
    return vectorizer

# Adapter le vectorizer en remplaçant None par une chaîne vide ""
name_vectorizer = vectorize_text('name')
ingredients_vectorizer = vectorize_text('ingredients')

# Fonction de préparation des données pour chaque lot
def prepare_batch(batch):
    inputs = {col: [] for col in categorical_columns}
    inputs['name'] = []
    inputs['ingredients'] = []
    outputs = []

    for entry in batch:
        for col in categorical_columns:
            value = entry.get(col, 'Unknown')  # Remplacer les None ou valeurs manquantes par 'Unknown'
            inputs[col].append(label_encoders[col].transform([value])[0])

        # Remplacer None par une chaîne vide
        inputs['name'].append(entry.get('name', '') if entry.get('name') is not None else "")
        inputs['ingredients'].append(entry.get('ingredients', '') if entry.get('ingredients') is not None else "")
        outputs.append(entry.get('ecoscore_score', 0.0))

    # Convertir les entrées textuelles en tensors
    inputs['name'] = name_vectorizer(tf.convert_to_tensor(inputs['name']))
    inputs['ingredients'] = ingredients_vectorizer(tf.convert_to_tensor(inputs['ingredients']))

    for col in categorical_columns:
        inputs[col] = tf.convert_to_tensor(inputs[col])

    return inputs, tf.convert_to_tensor(outputs)

# Fonction de génération de données par lots
def batch_generator(file, batch_size=32):
    batch = []
    for entry in load_jsonl(file):
        batch.append(entry)
        if len(batch) == batch_size:
            yield prepare_batch(batch)
            batch = []
    if batch:
        yield prepare_batch(batch)

# Construction du modèle
input_layers = []
embedding_layers = []

# Ajouter les entrées et embeddings pour les colonnes catégorielles
for col in categorical_columns:
    input_layer = Input(shape=(1,), name=col)
    embedding_layer = Embedding(input_dim=len(label_encoders[col].classes_), output_dim=10)(input_layer)
    embedding_layer = Reshape((10,))(embedding_layer)  # Assurez que chaque embedding est de forme (batch_size, 10)
    input_layers.append(input_layer)
    embedding_layers.append(embedding_layer)

# Entrées textuelles
name_input = Input(shape=(None,), dtype=tf.int64, name='name')
ingredients_input = Input(shape=(None,), dtype=tf.int64, name='ingredients')

# Appliquer l'Embedding après TextVectorization
name_embedding = Embedding(input_dim=20000, output_dim=10)(name_input)  # Assurez-vous d'avoir les bons input_dim et output_dim
ingredients_embedding = Embedding(input_dim=20000, output_dim=10)(ingredients_input)

name_embedding = GlobalAveragePooling1D()(name_embedding)
ingredients_embedding = GlobalAveragePooling1D()(ingredients_embedding)

input_layers.extend([name_input, ingredients_input])
embedding_layers.extend([name_embedding, ingredients_embedding])

# Fusionner toutes les couches
merged = concatenate(embedding_layers)
x = Dense(128, activation='relu')(merged)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='linear', name='ecoscore_score')(x)

# Créer le modèle
model = Model(inputs=input_layers, outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Entraînement du modèle
train_data = batch_generator(train_file, batch_size=32)
valid_data = batch_generator(valid_file, batch_size=32)

# Ajuster le modèle en utilisant les données par lots
model.fit(
    train_data,
    validation_data=valid_data,
    steps_per_epoch=2000,  # ajuster en fonction de la taille du fichier d'entraînement
    validation_steps=500,  # ajuster en fonction de la taille du fichier de validation
    epochs=10
)

# Évaluation sur les données de test
test_data = batch_generator(test_file, batch_size=32)
model.evaluate(test_data, steps=500)  # ajuster en fonction de la taille du fichier de test
