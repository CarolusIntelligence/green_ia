import numpy as np
import pandas as pd
import os
import warnings
import sys
import torch
from sklearn.feature_extraction.text import CountVectorizer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from io import StringIO

pd.set_option('display.max_rows', 100)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# declaration en globale des vectorizers
vectorizer_packaging = CountVectorizer()
vectorizer_name = CountVectorizer()
vectorizer_ingredients = CountVectorizer()
vectorizer_categories = CountVectorizer()

# pre-ajustement vectorizers sur echantillon de donnees
def fit_vectorizers(data_path, file_id, chunk_size):
    first_chunk = next(read_jsonl_in_chunks(data_path + file_id + "_train_01.jsonl", chunk_size))
    vectorizer_packaging.fit(first_chunk['packaging'].astype(str))
    vectorizer_name.fit(first_chunk['name'].astype(str))
    vectorizer_ingredients.fit(first_chunk['ingredients'].astype(str))
    vectorizer_categories.fit(first_chunk['categories'].astype(str))

class ComplexNN(nn.Module):
    def __init__(self, input_size):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def train_model(model, data_path, file_id, chunk_size, batch_size, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for x_batch, y_batch in data_generator(data_path, file_id, chunk_size, batch_size, "train"):
            optimizer.zero_grad()
            outputs = model(x_batch)
            y_batch = y_batch.view(-1, 1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            del x_batch, y_batch
            torch.cuda.empty_cache()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}')

# evaluation
def evaluate_model(model, data_path, file_id, chunk_size, batch_size, criterion, data_split="valid"):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in data_generator(data_path, file_id, chunk_size, batch_size, data_split):
            output = model(x_batch)
            predictions.extend(output.numpy().flatten()) 
            targets.extend(y_batch.numpy().flatten())
            # liberation memoire
            del x_batch, y_batch
            torch.cuda.empty_cache()
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')
    return mse, mae, r2

def data_generator(data_path, file_id, chunk_size, batch_size, data_split):
    file_path = f"{data_path}{file_id}_{data_split}.jsonl"
    for chunk in read_jsonl_in_chunks(file_path, chunk_size):
        x_data, y_data = data_to_tensor(chunk)
        dataset = TensorDataset(x_data, y_data)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for x_batch, y_batch in data_loader:
            yield x_batch, y_batch
        del x_data, y_data, dataset, data_loader
        torch.cuda.empty_cache()

def read_jsonl_in_chunks(file_path, chunk_size):
    chunks = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            chunks.append(pd.read_json(StringIO(line), lines=True))
            if (i + 1) % chunk_size == 0:
                yield pd.concat(chunks, ignore_index=True)
                chunks = []
    if chunks:
        yield pd.concat(chunks, ignore_index=True)

def data_to_tensor(chunk):
    y = torch.tensor(chunk['ecoscore_score'].values, dtype=torch.float32)
    chunk['packaging'] = chunk['packaging'].astype(str)
    chunk['name'] = chunk['name'].astype(str)
    chunk['ingredients'] = chunk['ingredients'].astype(str)
    chunk['categories'] = chunk['categories'].astype(str)
    encoded_packaging = vectorizer_packaging.transform(chunk['packaging']).toarray()
    encoded_name = vectorizer_name.transform(chunk['name']).toarray()
    encoded_ingredients = vectorizer_ingredients.transform(chunk['ingredients']).toarray()
    encoded_categories = vectorizer_categories.transform(chunk['categories']).toarray()
    x = pd.concat([
        chunk[['groups', 'countries', 'labels_note']].reset_index(drop=True),
        pd.DataFrame(encoded_packaging, columns=[f'packaging_{i}' for i in range(encoded_packaging.shape[1])]),
        pd.DataFrame(encoded_name, columns=[f'name_{i}' for i in range(encoded_name.shape[1])]),
        pd.DataFrame(encoded_ingredients, columns=[f'ingredients_{i}' for i in range(encoded_ingredients.shape[1])]),
        pd.DataFrame(encoded_categories, columns=[f'categories_{i}' for i in range(encoded_categories.shape[1])])
    ], axis=1)
    x = torch.tensor(x.values, dtype=torch.float32)
    return x, y



def main(chunk_size, file_id, data_path):
    chunk_size = int(chunk_size)
    batch_size = 64

    fit_vectorizers(data_path, file_id, chunk_size)
    first_chunk = next(read_jsonl_in_chunks(data_path + file_id + "_train_01.jsonl", chunk_size))
    x_sample, _ = data_to_tensor(first_chunk)
    input_size = x_sample.shape[1]  
    model = ComplexNN(input_size) 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, data_path, file_id, chunk_size, batch_size, criterion, optimizer, epochs=5)
    print("Validation Metrics:")
    evaluate_model(model, data_path, file_id, chunk_size, batch_size, criterion, data_split="valid")
    print("Test Metrics:")
    evaluate_model(model, data_path, file_id, chunk_size, batch_size, criterion, data_split="test")

if __name__ == "__main__":
    chunk_size = int(sys.argv[1])
    file_id = sys.argv[2]
    data_path = sys.argv[3]
    main(chunk_size, file_id, data_path)
