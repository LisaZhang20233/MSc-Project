import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare the dataset
class CodeDataset(Dataset):
    def __init__(self, texts, vectors, labels, tokenizer, max_length):
        self.texts = texts
        self.vectors = vectors
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        vector = self.vectors[index]
        label = self.labels[index]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Return the combined input and the label
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'vector': torch.tensor(vector, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Load the data
file_path = 'GNN/new_train_GIN.pkl'
data = pd.read_pickle(file_path)

# Label encode the category column
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['category'])

# Split the data
train_texts, val_texts, train_vectors, val_vectors, train_labels, val_labels = train_test_split(
    data['before_merge'].tolist(),
    data['GIN'].tolist(),
    labels,
    test_size=0.2,
    random_state=42
)

# Create dataset objects
train_dataset = CodeDataset(train_texts, train_vectors, train_labels, tokenizer, max_length=128)
val_dataset = CodeDataset(val_texts, val_vectors, val_labels, tokenizer, max_length=128)
# Load the RoBERTa model with classification head
class RobertaWithGIN(RobertaForSequenceClassification):
    def __init__(self, config, gin_vector_size, class_weights):
        super(RobertaWithGIN, self).__init__(config)
        self.gin_dense = torch.nn.Linear(gin_vector_size, config.hidden_size)
        self.classifier = torch.nn.Linear(config.hidden_size * 2, config.num_labels)
        self.class_weights = torch.tensor(class_weights).to(device)

    def forward(self, input_ids, attention_mask, vector, labels=None):
        # 将输入移动到GPU
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        vector = vector.to(device)

        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]  # 获取最后一层的隐藏状态，位于outputs[0]
        pooled_output = last_hidden_state[:, 0, :]  # 获取CLS token对应的hidden state

        gin_output = self.gin_dense(vector)

        combined_output = torch.cat((pooled_output, gin_output), dim=1)

        logits = self.classifier(combined_output)

        loss = None
        if labels is not None:
            labels = labels.to(device)  # 将标签移动到GPU
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}

# Initialize model with class weights
gin_vector_size = len(data['GIN'].iloc[0])
class_weights = [7.1343, 5.1598, 1.5014]  # 类别权重
model_config = RobertaConfig.from_pretrained('roberta-base', num_labels=len(set(labels)))
model = RobertaWithGIN(model_config, gin_vector_size, class_weights).to(device)  # 将模型移动到GPU

# Training arguments
training_args = TrainingArguments(
    output_dir='./RoBERTa_GNN_Version3/results',
    num_train_epochs=6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.1,
    logging_dir='./RoBERTa_GNN_Version3/logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,  # 如果支持混合精度，可以启用fp16来加速训练
)
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


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics  # Custom metrics function
)

# Train the model
trainer.train()

# Save the trained model

model.save_pretrained('./RoBERTa_GNN_Version3/model')
tokenizer.save_pretrained('./RoBERTa_GNN_Version3/tokenizer')

print("done")
