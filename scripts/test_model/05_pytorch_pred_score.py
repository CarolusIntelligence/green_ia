import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import re

# Configurations
train_file = "../data/07_data/07_train_01.jsonl"
valid_file = "../data/07_data/07_valid_01.jsonl"
test_file = "../data/07_data/07_test_01.jsonl"
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
early_stop_patience = 5
num_epochs = 50

# Utility function to clean and tokenize text
def tokenize_text(text):
    if text == "empty":
        return ["<empty>"]
    tokens = re.sub(r'\W+', ' ', text.lower()).split()
    return tokens

def collate_fn(batch):
    numeric_features = torch.stack([item[0]["numeric_features"] for item in batch])
    packaging = pad_sequence([item[0]["packaging"] for item in batch], batch_first=True)
    name = pad_sequence([item[0]["name"] for item in batch], batch_first=True)
    ingredients = pad_sequence([item[0]["ingredients"] for item in batch], batch_first=True)
    categories = pad_sequence([item[0]["categories"] for item in batch], batch_first=True)
    
    targets = torch.stack([item[1] for item in batch])
    
    return {
        "numeric_features": numeric_features,
        "packaging": packaging,
        "name": name,
        "ingredients": ingredients,
        "categories": categories
    }, targets

# Build vocabulary from dataset
def build_vocab(dataset, field):
    vocab = build_vocab_from_iterator(map(lambda x: tokenize_text(x[field]), dataset), specials=["<pad>", "<unk>", "<empty>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

# Dataset loader for JSONL files
class EcoscoreDataset(Dataset):
    def __init__(self, file_path, vocabs=None):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # Build vocabs if not provided
        if vocabs is None:
            self.vocabs = {
                "packaging": build_vocab(self.data, "packaging"),
                "name": build_vocab(self.data, "name"),
                "ingredients": build_vocab(self.data, "ingredients"),
                "categories": build_vocab(self.data, "categories")
            }
        else:
            self.vocabs = vocabs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Extract numerical features
        numeric_features = [
            sample["labels_note"], sample["countries"], sample["groups"]
        ]

        # Extract and tokenize text features
        packaging = torch.tensor(self.vocabs["packaging"].lookup_indices(tokenize_text(sample["packaging"])), dtype=torch.long)
        name = torch.tensor(self.vocabs["name"].lookup_indices(tokenize_text(sample["name"])), dtype=torch.long)
        ingredients = torch.tensor(self.vocabs["ingredients"].lookup_indices(tokenize_text(sample["ingredients"])), dtype=torch.long)
        categories = torch.tensor(self.vocabs["categories"].lookup_indices(tokenize_text(sample["categories"])), dtype=torch.long)

        target = sample["ecoscore_score"]
        return {
            "numeric_features": torch.tensor(numeric_features, dtype=torch.float32),
            "packaging": packaging,
            "name": name,
            "ingredients": ingredients,
            "categories": categories
        }, torch.tensor(target, dtype=torch.float32)

# LSTM model with text embeddings
class EcoscoreLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, vocab_sizes, embedding_dim=64):
        super(EcoscoreLSTM, self).__init__()
        
        # Embeddings for textual fields
        self.embedding_packaging = nn.EmbeddingBag(vocab_sizes["packaging"], embedding_dim, sparse=True)
        self.embedding_name = nn.EmbeddingBag(vocab_sizes["name"], embedding_dim, sparse=True)
        self.embedding_ingredients = nn.EmbeddingBag(vocab_sizes["ingredients"], embedding_dim, sparse=True)
        self.embedding_categories = nn.EmbeddingBag(vocab_sizes["categories"], embedding_dim, sparse=True)

        # LSTM layers
        self.lstm = nn.LSTM(input_size + 4 * embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        numeric_features = x["numeric_features"]
        packaging_embed = self.embedding_packaging(x["packaging"])
        name_embed = self.embedding_name(x["name"])
        ingredients_embed = self.embedding_ingredients(x["ingredients"])
        categories_embed = self.embedding_categories(x["categories"])
        
        # Concatenate all features
        combined_features = torch.cat((numeric_features, packaging_embed, name_embed, ingredients_embed, categories_embed), dim=1)
        combined_features = combined_features.unsqueeze(1)  # Add batch dimension

        lstm_out, _ = self.lstm(combined_features)
        output = self.fc(lstm_out[:, -1, :])  # Use the output of the last LSTM unit
        return output

# Metrics
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# Training loop with early stopping
def train_model(model, train_loader, valid_loader, optimizer, optimizer_sparse, criterion, num_epochs, early_stop_patience):
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            targets = targets.to(device)

            optimizer.zero_grad()  # Reset gradients for Adam
            optimizer_sparse.zero_grad()  # Reset gradients for SparseAdam
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()

            optimizer.step()  # Update Adam parameters
            optimizer_sparse.step()  # Update SparseAdam parameters
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        valid_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, targets in valid_loader:
                for k in inputs:
                    inputs[k] = inputs[k].to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                valid_loss += loss.item()
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(outputs.squeeze().cpu().numpy())

        valid_loss /= len(valid_loader)
        mae, rmse, r2 = compute_metrics(np.array(y_true), np.array(y_pred))

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}')

        # Early stopping
        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping")
                break

# Récupérer les prédictions du LSTM sur un loader
def get_lstm_predictions(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.squeeze().cpu().numpy())
    return np.array(y_true), np.array(y_pred)

# Charger les datasets
train_dataset = EcoscoreDataset(train_file)
valid_dataset = EcoscoreDataset(valid_file, train_dataset.vocabs)
test_dataset = EcoscoreDataset(test_file, train_dataset.vocabs)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Model instantiation
vocab_sizes = {k: len(v) for k, v in train_dataset.vocabs.items()}
input_size = 3  # Numerical features
hidden_size = 128
output_size = 1
num_layers = 2
dropout = 0.2

model = EcoscoreLSTM(input_size, hidden_size, output_size, num_layers, dropout, vocab_sizes)
model = model.to(device)

# Loss and optimizer with L2 regularization (weight_decay)
criterion = nn.MSELoss()

# Adam optimizer for non-sparse parameters
optimizer = optim.Adam([
    {'params': model.lstm.parameters()},
    {'params': model.fc.parameters()}
], lr=0.001, weight_decay=0.0001)  # Adding L2 regularization

# SparseAdam for sparse embeddings
optimizer_sparse = optim.SparseAdam([
    {'params': model.embedding_packaging.parameters()},
    {'params': model.embedding_name.parameters()},
    {'params': model.embedding_ingredients.parameters()},
    {'params': model.embedding_categories.parameters()}
], lr=0.001)

# Training the model
train_model(model, train_loader, valid_loader, optimizer, optimizer_sparse, criterion, num_epochs, early_stop_patience)

# Obtenir les prédictions du LSTM sur les données d'entraînement et de validation
train_true, train_pred_lstm = get_lstm_predictions(model, train_loader)
valid_true, valid_pred_lstm = get_lstm_predictions(model, valid_loader)

# Utiliser les prédictions de l'LSTM comme features supplémentaires pour le Ridge Regression (Stacking)
train_features = np.column_stack([train_pred_lstm])
valid_features = np.column_stack([valid_pred_lstm])

# Instancier le modèle Ridge Regression pour le stacking
ridge_model = Ridge(alpha=1.0)

# Entraîner le modèle sur les données de train
ridge_model.fit(train_features, train_true)

# Obtenir les prédictions du Ridge Regression sur les données de validation
valid_pred_ridge = ridge_model.predict(valid_features)

# Combiner les prédictions LSTM et Ridge Regression (ici par moyenne pondérée)
combined_pred = 0.7 * valid_pred_lstm + 0.3 * valid_pred_ridge

# Calculer les métriques sur les prédictions combinées
mae_combined, rmse_combined, r2_combined = compute_metrics(valid_true, combined_pred)

print(f'Validation MAE (Combiné): {mae_combined:.4f}, RMSE (Combiné): {rmse_combined:.4f}, R2 (Combiné): {r2_combined:.4f}')
