import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from torch.nn.utils.rnn import pad_sequence
import re
from torchtext.vocab import build_vocab_from_iterator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Configurations
valid_file = "../data/07_data/07_valid_01.jsonl"  # Fichier de validation
model_path = '../best_models/best_model.ci'  # Modèle enregistré
vocab_path = '../best_models/vocabs.json'  # Vocabulaires enregistrés
output_csv = "validation_predictions_combined.csv"  # Fichier CSV de sortie
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Utility function to clean and tokenize text
def tokenize_text(text):
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

# Dataset loader for JSONL files
class EcoscoreDataset(Dataset):
    def __init__(self, file_path, vocabs):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
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

# Load the saved vocabs
with open(vocab_path, 'r') as f:
    vocabs_data = json.load(f)

# Recreate vocabs
vocabs = {key: build_vocab_from_iterator([[token] for token in tokens], specials=["<pad>", "<unk>"])
          for key, tokens in vocabs_data.items()}

for vocab in vocabs.values():
    vocab.set_default_index(vocab["<unk>"])

# Load the validation dataset
valid_dataset = EcoscoreDataset(valid_file, vocabs=vocabs)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Load vocab sizes from saved vocabs
vocab_sizes = {k: len(v) for k, v in vocabs.items()}

# Model instantiation
input_size = 3  # Numerical features
hidden_size = 128
output_size = 1
num_layers = 2
dropout = 0.2

model = EcoscoreLSTM(input_size, hidden_size, output_size, num_layers, dropout, vocab_sizes)
model = model.to(device)

# Load the best model
model.load_state_dict(torch.load(model_path))

# Evaluate on validation set and save predictions
model.eval()
y_true, y_pred_lstm = [], []
original_data = []

with torch.no_grad():
    for inputs, targets in valid_loader:
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        y_true.extend(targets.cpu().numpy())
        y_pred_lstm.extend(outputs.squeeze().cpu().numpy())

        # Save original inputs for output to CSV
        original_data.extend(inputs["numeric_features"].cpu().numpy())

# Étape complémentaire: Utiliser Gradient Boosting pour affiner les prédictions
train_features = np.column_stack([y_pred_lstm])  # Utiliser les prédictions LSTM comme feature

# Instancier et entraîner le modèle GradientBoostingRegressor
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(train_features, y_true)

# Obtenir les prédictions du Gradient Boosting
y_pred_gb = gb_model.predict(train_features)

# Combiner les prédictions LSTM et Gradient Boosting (moyenne simple)
combined_pred = 0.5 * np.array(y_pred_lstm) + 0.5 * np.array(y_pred_gb)

# Calculer les métriques finales
mae_combined = mean_absolute_error(y_true, combined_pred)
rmse_combined = np.sqrt(mean_squared_error(y_true, combined_pred))
r2_combined = r2_score(y_true, combined_pred)

print(f'Validation MAE (Combiné): {mae_combined:.4f}, RMSE (Combiné): {rmse_combined:.4f}, R2 (Combiné): {r2_combined:.4f}')

# Prepare data for output
df = pd.DataFrame(original_data, columns=["labels_note", "countries", "groups"])
df["true"] = y_true
df["predictions_lstm"] = y_pred_lstm
df["predictions_gb"] = y_pred_gb
df["predictions_combined"] = combined_pred

# Save predictions to CSV
df.to_csv(output_csv, index=False)

print(f"Combined predictions saved to {output_csv}")
