import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# 1. Load the Dataset
url = 'https://raw.githubusercontent.com/Polar-SemEval/trial-data/refs/heads/main/Trial_Data.csv'
try:
    df = pd.read_csv(url)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Select only relevant columns
df = df[['text', 'polarization']]

# 2. Preprocessing & Split
# Ensure text is string and labels are integers
df['text'] = df['text'].astype(str)
df['polarization'] = df['polarization'].astype(int)

# Split into training and validation sets (80% train, 20% val)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), 
    df['polarization'].tolist(), 
    test_size=0.2, 
    random_state=42
)

# 3. Tokenizer & Dataset Class
# Use multilingual BERT tokenizer
model_name = 'bert-base-multilingual-uncased' 
tokenizer = BertTokenizer.from_pretrained(model_name)

class PolarizationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = PolarizationDataset(train_texts, train_labels, tokenizer)
val_dataset = PolarizationDataset(val_texts, val_labels, tokenizer)

# 4. Model Initialization
# Load pre-trained BERT with a classification head on top (num_labels=2 for binary)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 5. Training Setup
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,              # Small dataset, 3-5 epochs usually enough
    per_device_train_batch_size=8,   # Adjust based on your VRAM
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",           # Evaluate at end of every epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"                 # Disable wandb/mlflow logging for simple run
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 6. Train and Evaluate
print("Starting training...")
trainer.train()

print("\nFinal Evaluation on Validation Set:")
eval_results = trainer.evaluate()
print(eval_results)

# 7. (Optional) Save the model
model.save_pretrained("./polarization_model")
tokenizer.save_pretrained("./polarization_model")