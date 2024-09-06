from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

# Load the data
file_path = 'dataset_new/train_merge.pkl'
data = pd.read_pickle(file_path)

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenize the 'before_merge' column
tokens = tokenizer.batch_encode_plus(data['before_merge'].tolist(), add_special_tokens=True, truncation=True, max_length=128)

# Encode the 'category' column
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['category'])

# Weight loss function
class_weights = torch.tensor([0.6733, 0.1886, 0.1381], dtype=torch.float32).to('cuda')  # Move class weights to GPU
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# Convert tokens and labels to lists for further processing
tokens_list = tokens['input_ids']
labels_list = labels.tolist()

# Split data into training and validation sets
train_tokens, val_tokens, train_labels, val_labels = train_test_split(tokens_list, labels_list, test_size=0.2, random_state=42)

class CodeDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.tokens[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Create Dataset objects
train_dataset = CodeDataset(train_tokens, train_labels)
val_dataset = CodeDataset(val_tokens, val_labels)

# Define the model and move it to GPU
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_encoder.classes_)).to('cuda')

# Custom Trainer to include weighted loss
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").to('cuda')  # Move labels to GPU
        inputs = {k: v.to('cuda') for k, v in inputs.items()}  # Ensure all inputs are moved to GPU
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Define custom metrics function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    # Compute multi-class accuracy
    accuracy = accuracy_score(labels, preds)

    # Compute macro-averaged F1 score
    macro_f1 = f1_score(labels, preds, average='macro')

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1
    }

# Define training arguments
training_args = TrainingArguments(
    # output_dir='./results',
    # num_train_epochs=3,
    # per_device_train_batch_size=8,
    # per_device_eval_batch_size=8,
    # weight_decay=0.1,
    # logging_dir='./logs',
    # logging_strategy="epoch",
    # logging_steps=10,
    # eval_strategy="epoch",
    # save_strategy="epoch",
    # fp16=True
    output_dir = './results',
    max_steps = 500000,  # 设置为500K步
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    learning_rate = 2e-5,
    warmup_steps = 10000,  # 热身步数，可以相应增加
    weight_decay = 0.01,
    logging_dir = './logs',
    logging_steps = 500,  # 日志记录的步数间隔
    save_steps = 10000,  # 保存模型的步数间隔
    evaluation_strategy = "steps",  # 评估策略
    eval_steps = 10000,  # 评估的步数间隔
    fp16 = True # 如果可用，使用半精度浮点数

)

# Data collator to handle padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize the custom trainer
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics  # Custom metrics function
)

# Start training
trainer.train()

# Evaluate the model
eval_result = trainer.evaluate()

# Print evaluation results
print(f"Evaluation Results: {eval_result}")

# Save the model and tokenizer
model.save_pretrained('./model')
tokenizer.save_pretrained('./tokenizer')
