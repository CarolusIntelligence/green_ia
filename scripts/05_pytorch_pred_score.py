import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import os
import warnings
import json
import re
import sys

pd.set_option('display.max_rows', 100)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
pd.set_option('future.no_silent_downcasting', True)



###############################################################################
# MAIN ########################################################################
###############################################################################
def main(chunk_size, file_id, data_path, MAX_SEQ_LEN, batch_size, embed_dim, hidden_dim, lr, patience, best_model_path):
    chunk_size = int(chunk_size)
    MAX_SEQ_LEN = int(MAX_SEQ_LEN)
    batch_size = int(batch_size)
    embed_dim = int(embed_dim)
    hidden_dim = int(hidden_dim)
    lr = float(lr)
    patience = int(patience)
    valid = data_path + file_id + '_valid.jsonl' 

    def load_jsonl(file_path):
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
        return pd.DataFrame(data)
    
    train_df = load_jsonl(data_path + file_id + '_train_01.jsonl') 
    test_df = load_jsonl(data_path + file_id + '_test_01.jsonl') 
    X_train = train_df.drop(columns=['ecoscore_score'])
    y_train = train_df['ecoscore_score']
    X_test = test_df.drop(columns=['ecoscore_score'])
    y_test = test_df['ecoscore_score']
    num_cols = ['groups', 'countries', 'labels_note']
    text_cols = ['packaging', 'name', 'ingredients', 'categories']
    X_train_num = X_train[num_cols]
    X_train_text = X_train[text_cols].astype(str)  
    X_test_num = X_test[num_cols]
    X_test_text = X_test[text_cols].astype(str)    
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_test_num = scaler.transform(X_test_num)
    tokenizer = get_tokenizer('basic_english')

    def tokenize(text):
        return tokenizer(text)

    def build_vocab(texts):
        vocab = build_vocab_from_iterator(map(tokenize, texts))
        vocab.insert_token('<unk>', 0)  
        vocab.insert_token('<pad>', 1)  
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    all_texts = pd.concat([X_train_text[col] for col in text_cols])
    vocab = build_vocab(all_texts)

    def text_to_indices(text):
        tokens = tokenize(text)
        indices = [vocab[token] for token in tokens]
        if len(indices) < MAX_SEQ_LEN:
            indices += [vocab['<pad>']] * (MAX_SEQ_LEN - len(indices)) 
        else:
            indices = indices[:MAX_SEQ_LEN]  
        return torch.tensor(indices)

    def text_data_to_tensor(text_data):
        return torch.stack([text_to_indices(text) for text in text_data])

    X_train_text_combined = X_train_text.apply(lambda x: ' '.join(x), axis=1)
    X_test_text_combined = X_test_text.apply(lambda x: ' '.join(x), axis=1)
    X_train_text_indices = text_data_to_tensor(X_train_text_combined)
    X_test_text_indices = text_data_to_tensor(X_test_text_combined)

    class CustomDataset(Dataset):
        def __init__(self, num_data, text_data, labels):
            self.num_data = torch.tensor(num_data, dtype=torch.float)
            self.text_data = text_data
            self.labels = torch.tensor(labels.values, dtype=torch.float)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return self.num_data[idx], self.text_data[idx], self.labels[idx]

    train_dataset = CustomDataset(X_train_num, X_train_text_indices, y_train)
    test_dataset = CustomDataset(X_test_num, X_test_text_indices, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    class TextEncoder(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim):
            super(TextEncoder, self).__init__()
            self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
            self.fc = nn.Linear(embed_dim, hidden_dim)
        def forward(self, x):
            x = self.embedding(x)
            x = torch.relu(self.fc(x))
            return x

    class ComplexModel(nn.Module):
        def __init__(self, vocab_size, num_numeric_features, embed_dim, hidden_dim, output_dim):
            super(ComplexModel, self).__init__()
            self.text_encoder = TextEncoder(vocab_size, embed_dim, hidden_dim)
            self.numeric_fc = nn.Linear(num_numeric_features, hidden_dim)
            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        def forward(self, numeric_data, text_data):
            text_features = self.text_encoder(text_data)
            numeric_features = torch.relu(self.numeric_fc(numeric_data))
            combined_features = torch.cat((text_features, numeric_features), dim=1)
            x = torch.relu(self.fc1(combined_features))
            x = self.fc2(x)
            return x

    vocab_size = len(vocab)
    num_numeric_features = len(num_cols)
    output_dim = 1

    model = ComplexModel(vocab_size, num_numeric_features, embed_dim, hidden_dim, output_dim)
    sparse_params = list(model.text_encoder.embedding.parameters())  
    dense_params = list(set(model.parameters()) - set(sparse_params))
    optimizer_sparse = optim.SparseAdam(sparse_params, lr=lr) 
    optimizer_dense = optim.Adam(dense_params, lr=lr) 
    criterion = nn.MSELoss()

    def calculate_metrics(y_true, y_pred):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, r2, mae

    def train(model, dataloader, criterion, optimizer_sparse, optimizer_dense, device):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        for num_data, text_data, labels in dataloader:
            num_data, text_data, labels = num_data.to(device), text_data.to(device), labels.to(device)
            optimizer_sparse.zero_grad()
            optimizer_dense.zero_grad()
            outputs = model(num_data, text_data)
            outputs = outputs.squeeze()
            labels = labels.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_sparse.step()
            optimizer_dense.step()
            running_loss += loss.item() * num_data.size(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            all_labels.append(labels)
            all_preds.append(outputs)
        epoch_loss = running_loss / len(dataloader.dataset)
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        rmse, r2, mae = calculate_metrics(all_labels, all_preds)
        return epoch_loss, rmse, r2, mae

    def evaluate(model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for num_data, text_data, labels in dataloader:
                num_data, text_data, labels = num_data.to(device), text_data.to(device), labels.to(device)
                outputs = model(num_data, text_data)
                outputs = outputs.squeeze()  
                labels = labels.squeeze()  
                loss = criterion(outputs, labels)
                running_loss += loss.item() * num_data.size(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                all_labels.append(labels)
                all_preds.append(outputs)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        rmse, r2, mae = calculate_metrics(all_labels, all_preds)
        return epoch_loss, rmse, r2, mae


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    num_epochs = 2  
    best_loss = float('inf')
    trigger_times = 0

    for epoch in range(num_epochs):
        train_loss, train_rmse, train_r2, train_mae = train(model, train_loader, criterion, optimizer_sparse, optimizer_dense, device)
        test_loss, test_rmse, test_r2, test_mae = evaluate(model, test_loader, criterion, device)
        print(f'epoch {epoch+1}/{num_epochs}')
        print(f'train loss: {train_loss:.4f}, train RMSE: {train_rmse:.4f}, train R2: {train_r2:.4f}, train MAE: {train_mae:.4f}')
        print(f'test loss: {test_loss:.4f}, test RMSE: {test_rmse:.4f}, test R2: {test_r2:.4f}, test MAE: {test_mae:.4f}')
        if test_loss < best_loss:
            best_loss = test_loss
            trigger_times = 0
            torch.save(model.state_dict(), best_model_path)
            print(f'model improved and saved at epoch {epoch+1}')
        else:
            trigger_times += 1
            print(f'no improvement in test loss for {trigger_times} epoch')
        if trigger_times >= patience:
            print('early stopping')
            break

    model.load_state_dict(torch.load(best_model_path))
    print('best model loaded for evaluation')


if __name__ == "__main__":
    chunk_size = sys.argv[1]
    file_id = sys.argv[2]
    data_path = sys.argv[3]
    MAX_SEQ_LEN = sys.argv[4]
    batch_size = sys.argv[5]
    embed_dim = sys.argv[6]
    hidden_dim = sys.argv[7]
    lr = sys.argv[8]
    patience = sys.argv[9]
    best_model_path = sys.argv[10]

    main(chunk_size, file_id, data_path, MAX_SEQ_LEN, batch_size, embed_dim, hidden_dim, lr, patience, best_model_path)