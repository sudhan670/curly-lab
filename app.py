import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from transformers import TrainingArguments, Trainer
import numpy as np 
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

data = load_data('data1.csv')
print(data)
st.write("Data Columns:", data.columns)
st.write("Data Head:")
st.write(data.head())
expected_columns = ['query', 'category']
actual_columns = data.columns.tolist()
print(actual_columns)

# Preprocess data
def preprocess_data(data):
    # Rename columns to lowercase
    data.columns = [col.lower() for col in data.columns]
    if 'query' not in data.columns or 'category' not in data.columns:
        raise KeyError("Columns 'query' and 'category' are not found in the DataFrame.")
    return data[['query', 'category']]
     

data = preprocess_data(data)
st.write("Preprocessed Data:", data.head())
     

# Encode labels
label_encoder = LabelEncoder()
data['category'] = label_encoder.fit_transform(data['category'])
     

# Split data
X = data['query'].tolist()
y = data['category'].tolist()
     

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# Tokenize data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

def tokenize_data(data, tokenizer):
    return tokenizer(data, padding=True, truncation=True, max_length=512, return_tensors='pt')

X_train_tokenized = tokenize_data(X_train, tokenizer)
X_val_tokenized = tokenize_data(X_val, tokenizer)

     

# Define dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

     

# Define training arguments and trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

     

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
st.write("Evaluation Results:", eval_results)
     
