import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
from transformers import DistilBertModel, DistilBertTokenizer

class EcoScoreDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        numeric_data = torch.tensor([sample['pnns_1'], sample['countries'], sample['nova'], sample['palm_oil'], 
                                     sample['nutriscore_tags'], sample['additives']]).float()
        text_data = f"{sample['name']} {sample['ecoscore_data']} {sample['food_group']} {sample['nutrient_level']} " \
                    f"{sample['categories']} {sample['stores']} {sample['main_category']} {sample['keywords']} " \
                    f"{sample['packaging']} {sample['ingredients']}"
        label = torch.tensor(sample['ecoscore_tags']).float()
        
        return numeric_data, text_data, label

class HybridModel(nn.Module):
    def __init__(self, num_features, dropout=0.5):
        super(HybridModel, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_embedding_dim = self.bert.config.hidden_size
        
        self.fc = nn.Sequential(
            nn.Linear(32 + self.text_embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  
        )
    
    def forward(self, numeric_data, text_data):
        numeric_out = self.mlp(numeric_data)
        
        encoded_inputs = self.bert_tokenizer(text_data, return_tensors='pt', padding=True, truncation=True, max_length=128).to(numeric_data.device)
        text_out = self.bert(**encoded_inputs).last_hidden_state[:, 0, :]  
        
        combined = torch.cat((numeric_out, text_out), dim=1)
        output = self.fc(combined)
        
        return output

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def test_model(model, test_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            numeric_data, text_data, _ = batch
            numeric_data = numeric_data.to(device)
            preds = model(numeric_data, text_data).squeeze().cpu().numpy()
            predictions.extend(preds.tolist())
    return predictions

def save_predictions(data, predictions, output_file):
    for i, pred in enumerate(predictions):
        data[i]['predicted_ecoscore_tags'] = pred
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def train(model, train_loader, val_loader, epochs=20, lr=1e-5, save_path="best_model_01.ci"):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_loss = float('inf') 
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            numeric_data, text_data, labels = batch
            
            numeric_data, labels = numeric_data.to(device), labels.to(device)
            
            predictions = model(numeric_data, text_data)
            loss = criterion(predictions.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader)}")
        
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss) 
        
        if val_loss < best_val_loss:
            print(f"Meilleure Validation Loss: {val_loss}, sauvegarde du modèle.")
            best_val_loss = val_loss
            save_model(model, save_path)
    
    print(f"Meilleur modèle sauvegardé avec une perte de validation: {best_val_loss}")

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            numeric_data, text_data, labels = batch
            numeric_data, labels = numeric_data.to(device), labels.to(device)
            
            predictions = model(numeric_data, text_data)
            loss = criterion(predictions.squeeze(), labels)
            val_loss += loss.item()
    
    print(f"Validation Loss: {val_loss / len(val_loader)}")
    return val_loss

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

if __name__ == "__main__":
    train_data = load_jsonl("../../data/05_data/05_train_02.jsonl")
    test_data = load_jsonl("../../data/05_data/05_test_02.jsonl")
    valid_data = load_jsonl("../../data/05_data/05_valid_02.jsonl")
    
    train_dataset = EcoScoreDataset(train_data)
    test_dataset = EcoScoreDataset(test_data)
    valid_dataset = EcoScoreDataset(valid_data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    
    num_features = 6  
    model = HybridModel(num_features, dropout=0.5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(model, train_loader, valid_loader, epochs=10, lr=1e-5, save_path="best_model_01.ci")
    
    model = load_model(model, "best_model_01.ci")
    
    test_predictions = test_model(model, test_loader, device)
    save_predictions(test_data, test_predictions, "test_pred_01.jsonl")
    
    valid_predictions = test_model(model, valid_loader, device)
    save_predictions(valid_data, valid_predictions, "valid_pred_01.jsonl")
