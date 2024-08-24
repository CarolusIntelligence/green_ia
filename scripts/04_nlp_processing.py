import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re
import sys
import os

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"file deleted: {file_path}")
    else:
        print(f"ERROR, does not exists: {file_path}")

def clean_text(text):
    if not text:
        return ""
    text = text.lower()  # convertir en minuscule
    text = re.sub(r'\b\d+\b', '', text)  # supprimer les chiffres isolés
    text = re.sub(r'[^\w\s]', '', text)  # supprimer les ponctuations
    text = ' '.join(word for word in text.split() if word not in stop_words)  # supprimer les mots vides
    return text

# extraction mots-clés avec TF-IDF pour chaque ligne
def extract_keywords(text_series):
    vectorizer = TfidfVectorizer(max_features=7)  # limite 7 mots-clés les plus importants
    X = vectorizer.fit_transform(text_series)
    feature_names = vectorizer.get_feature_names_out()
    dense_matrix = X.todense().tolist()
    keywords = []
    for row in dense_matrix:
        word_scores = {word: score for word, score in zip(feature_names, row) if score > 0}
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        keywords.append(' '.join([word for word, score in sorted_words]))
    return keywords

def processing(jsonl_04, jsonl_05, chunk_size): 
    with open(jsonl_05, 'w') as outfile:
        with open(jsonl_04, 'r') as infile:
            batch = []
            for line in infile:
                try:
                    data = json.loads(line.strip())
                    batch.append(data)
                except json.JSONDecodeError:
                    continue  
                if len(batch) >= chunk_size:
                    df = pd.DataFrame(batch)
                    for column in ['packaging', 'ingredients', 'categories']:
                        if column in df.columns:
                            df[column] = df[column].astype(str).apply(clean_text)
                            df[column] = extract_keywords(df[column])
                    df.to_json(outfile, orient='records', lines=True)
                    batch = []
            if batch:
                df = pd.DataFrame(batch)
                for column in ['packaging', 'name', 'ingredients', 'categories']:
                    if column in df.columns:
                        df[column] = df[column].astype(str).apply(clean_text)
                        df[column] = extract_keywords(df[column])
                df.to_json(outfile, orient='records', lines=True)



###############################################################################
# MAIN ########################################################################
###############################################################################
def main(chunk_size, file_id, data_path):
    chunk_size = int(chunk_size)
    jsonl_04 = data_path + file_id + '_openfoodfacts_04.jsonl' 
    jsonl_05 = data_path + file_id + '_openfoodfacts_05.jsonl' 
    print("start nlp processing")
    processing(jsonl_04, jsonl_05, chunk_size)
    #print("deleting file jsonl 04")
    #delete_file(jsonl_04)

if __name__ == "__main__":
    file_id = sys.argv[1]
    data_path = sys.argv[2]
    chunk_size = sys.argv[3]
    main(chunk_size, file_id, data_path)