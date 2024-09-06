import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import RobertaTokenizer, RobertaConfig
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载tokenizer和模型
tokenizer = RobertaTokenizer.from_pretrained('./RoBERTa_GNN_Version2/tokenizer')
model_config = RobertaConfig.from_pretrained('./RoBERTa_GNN_Version2/model')
gin_vector_size = 10 # 请使用训练时的GIN向量维度
class_weights = [0.6733, 0.1886, 0.1381]  # 使用训练时的类别权重

# 定义自定义模型类（与训练时相同）
class RobertaWithGIN(RobertaForSequenceClassification):
    def __init__(self, config, gin_vector_size, class_weights):
        super(RobertaWithGIN, self).__init__(config)
        self.gin_dense = torch.nn.Linear(gin_vector_size, config.hidden_size)
        self.classifier = torch.nn.Linear(config.hidden_size * 2, config.num_labels)
        self.class_weights = torch.tensor(class_weights).to(device)

    def forward(self, input_ids, attention_mask, vector, labels=None):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        vector = vector.to(device)

        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0, :]  # 使用CLS token对应的hidden state

        gin_output = self.gin_dense(vector)
        combined_output = torch.cat((pooled_output, gin_output), dim=1)

        logits = self.classifier(combined_output)

        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}

# 初始化模型
model = RobertaWithGIN(model_config, gin_vector_size, class_weights).to(device)
model = RobertaWithGIN.from_pretrained(
    './RoBERTa_GNN_Version2/model',
    config=model_config,
    gin_vector_size=gin_vector_size,
    class_weights=class_weights,
    ignore_mismatched_sizes=True  # 忽略不匹配的权重尺寸
)
model.to(device)

# 准备测试数据集
class TestCodeDataset(Dataset):
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

        # 对文本进行分词
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

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'vector': torch.tensor(vector, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 加载测试数据集
test_file_path = 'GNN/new_test_GIN.pkl'
test_data = pd.read_pickle(test_file_path)

# 将标签从字符串转换为数值
label_encoder = LabelEncoder()
test_labels = label_encoder.fit_transform(test_data['category'])

# 准备测试数据集
test_dataset = TestCodeDataset(
    test_data['before_merge'].tolist(),
    test_data['GIN'].tolist(),
    test_labels,  # 使用转换后的数值标签
    tokenizer,
    max_length=128
)

# DataLoader用于批量处理
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 评估函数
def evaluate_accuracy(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            vector = batch['vector'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, vector=vector)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

# 在测试集上评估准确率
accuracy = evaluate_accuracy(model, test_loader, device)

# 打印准确率
print(f"Test Accuracy: {accuracy * 100:.2f}%")
