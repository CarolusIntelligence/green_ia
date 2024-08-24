import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import layers, models, optimizers
from transformers import DistilBertTokenizer, TFDistilBertModel
from io import StringIO
import sys

# Set mixed precision to reduce memory usage
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Config GPU
def setup_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU configuration successful.")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")

setup_gpu()

# Read JSONL in batches
def read_jsonl_in_batches(file_path, batch_size):
    batch = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            try:
                json_data = pd.read_json(StringIO(line), lines=True)
                batch.append(json_data)
            except ValueError as ve:
                print(f"Error reading JSON line {i}: {ve}")
                continue

            if (i + 1) % batch_size == 0:
                yield pd.concat(batch, ignore_index=True)
                batch = []

        if batch:
            yield pd.concat(batch, ignore_index=True)

# Load data in batches
def load_data_in_batches(train_path, test_path, valid_path, batch_size=1000):
    train_batches = read_jsonl_in_batches(train_path, batch_size)
    test_batches = read_jsonl_in_batches(test_path, batch_size)
    valid_batches = read_jsonl_in_batches(valid_path, batch_size)
    return train_batches, test_batches, valid_batches

# Build model
def build_model(num_textual_features, num_numeric_features):
    distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    input_ids = layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')
    
    distilbert_outputs = distilbert_model(input_ids, attention_mask=attention_mask)[0][:, 0, :]  # Use pooled output for classification
    
    input_numeric = layers.Input(shape=(num_numeric_features,), dtype=tf.float32, name='numeric_input')
    
    concatenated = layers.Concatenate()([distilbert_outputs, input_numeric])
    
    dense1 = layers.Dense(128, activation='relu')(concatenated)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    output = layers.Dense(1, activation='linear')(dense2)
    
    model = models.Model(inputs=[input_ids, attention_mask, input_numeric], outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Training function
def training(train_batches, test_batches):
    label_encoders = {}
    is_first_batch = True
    num_textual_features = 1  
    num_numeric_features = 4
    scaler = StandardScaler()
    model = build_model(num_textual_features, num_numeric_features)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Collect all categories from training data
    all_categories = {}

    # Fit LabelEncoders on the entire dataset
    for batch in train_batches:
        # Check and prepare columns
        #required_columns = ['groups', 'packaging', 'name', 'countries', 'ingredients', 'categories', 'labels_note']
        required_columns = ['groups', 'name', 'categories', 'labels_note']
        available_columns = [col for col in required_columns if col in batch.columns]
        missing_columns = set(required_columns) - set(available_columns)

        if missing_columns:
            print(f"Warning: Missing columns in batch: {missing_columns}")

        # Handling missing columns by filling with NaN or a default value
        for col in missing_columns:
            batch[col] = 'Unknown' if col == 'name' else 0

        x = batch[available_columns]

        # Collect categories from training data
        for column in x.select_dtypes(include=['object']).columns:
            if column not in all_categories:
                all_categories[column] = set(x[column].astype(str).unique())
            else:
                all_categories[column].update(x[column].astype(str).unique())

    # Initialize LabelEncoders with all collected categories
    for column, categories in all_categories.items():
        label_encoders[column] = LabelEncoder()
        label_encoders[column].fit(list(categories))

    # Rewind the train_batches generator
    train_batches = read_jsonl_in_batches(train_path, batch_size=16)

    for batch in train_batches:
        y = batch['ecoscore_score'].values

        # Handling missing columns by filling with NaN or a default value
        for col in missing_columns:
            batch[col] = 'Unknown' if col == 'name' else 0

        x = batch[available_columns]
        
        # Encode categorical columns
        for column in x.select_dtypes(include=['object']).columns:
            x.loc[:, column] = x[column].astype(str)
            # Transform and handle unseen labels
            try:
                x.loc[:, column] = label_encoders[column].transform(x[column])
            except ValueError as e:
                print(f"Error in encoding column {column}: {e}")
                # Assign unknown category (-1)
                x.loc[:, column] = x[column].map(lambda s: -1 if s not in label_encoders[column].classes_ else s)

        x_scaled = scaler.fit_transform(x)

        # Tokenize text data
        text_data = x['name'].astype(str).tolist()
        encoded_inputs = tokenizer(text_data, padding='max_length', truncation=True, max_length=128, return_tensors='tf')

        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']

        print(f"input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}")

        x_numeric = x_scaled[:, num_textual_features:]  
        
        # Ensure the numeric data has the expected shape
        if x_numeric.shape[1] != num_numeric_features:
            print(f"Adjusting numeric data shape from {x_numeric.shape[1]} to {num_numeric_features}")
            x_numeric = np.pad(x_numeric, ((0, 0), (0, num_numeric_features - x_numeric.shape[1])), 'constant')

        x_train_numeric = tf.convert_to_tensor(x_numeric, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y, dtype=tf.float32)
        
        model.train_on_batch([input_ids, attention_mask, x_train_numeric], y_train)

    # Evaluate on test batches
    for batch in test_batches:
        batch.fillna('Unknown', inplace=True)
        y = batch['ecoscore_score'].values
        
        available_columns = [col for col in required_columns if col in batch.columns]
        
        for col in missing_columns:
            batch[col] = 'Unknown' if col == 'name' else 0

        x = batch[available_columns]
        
        for column in x.select_dtypes(include=['object']).columns:
            x.loc[:, column] = x[column].astype(str)
            try:
                x.loc[:, column] = label_encoders[column].transform(x[column])
            except ValueError as e:
                print(f"Error in encoding column {column}: {e}")
                # Assign unknown category (-1)
                x.loc[:, column] = x[column].map(lambda s: -1 if s not in label_encoders[column].classes_ else s)
        x_scaled = scaler.transform(x)

        text_data = x['name'].astype(str).tolist()
        encoded_inputs = tokenizer(text_data, padding='max_length', truncation=True, max_length=128, return_tensors='tf')

        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']

        x_numeric = x_scaled[:, num_textual_features:]  

        if x_numeric.shape[1] != num_numeric_features:
            print(f"Adjusting numeric data shape from {x_numeric.shape[1]} to {num_numeric_features}")
            x_numeric = np.pad(x_numeric, ((0, 0), (0, num_numeric_features - x_numeric.shape[1])), 'constant')

        x_test_numeric = tf.convert_to_tensor(x_numeric, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y, dtype=tf.float32)

        loss, mae = model.evaluate([input_ids, attention_mask, x_test_numeric], y_test)
        print(f"Test Loss: {loss}, Test MAE: {mae}")

# Main function
def main(file_id, data_path):
    global train_path, batch_size
    train_path = data_path + file_id + "_train" + ".jsonl"
    test = data_path + file_id + "_test" + ".jsonl"
    valid = data_path + file_id + "_valid" + ".jsonl"
    print("Setting up GPU")
    setup_gpu()
    train_batches, test_batches, valid_batches = load_data_in_batches(train_path, test, valid, batch_size=16)  # Reduce batch size to lower memory usage
    training(train_batches, test_batches)

if __name__ == "__main__":
    file_id = sys.argv[1]
    data_path = sys.argv[2]
    main(file_id, data_path)
