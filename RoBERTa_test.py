import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# 加载测试集
test_file_path = 'dataset_new/new_valid.pkl'
test_data = pd.read_pickle(test_file_path)

# 加载tokenizer和模型
tokenizer = RobertaTokenizer.from_pretrained('./merge/RoBERTa_version1/tokenizer')  # 使用已保存的tokenizer
model = RobertaForSequenceClassification.from_pretrained('./merge/RoBERTa_version1/model')  # 使用已保存的模型
model.to('cuda')  # 将模型移动到GPU

# Tokenize 测试数据
test_tokens = test_data['before_merge'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=128))

# 假设你的标签已经编码为数值类型，如果标签是字符串，需要先将其转换为数值
label_encoder = LabelEncoder()
test_labels = label_encoder.fit_transform(test_data['category'].tolist())  # 将字符串标签转换为整数

# 创建测试集数据集对象
class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.tokens[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)  # 确保标签是整数类型
        }

test_dataset = CodeDataset(test_tokens.tolist(), test_labels)

# 使用模型进行预测
model.eval()  # 设置模型为评估模式
predictions, true_labels = [], []

with torch.no_grad():
    for data in test_dataset:
        inputs = data['input_ids'].unsqueeze(0).to('cuda')  # 增加批次维度，并移动到GPU
        labels = data['labels'].unsqueeze(0).to('cuda')
        outputs = model(inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1)
        predictions.append(predicted_class.item())
        true_labels.append(labels.item())

# 计算准确率或其他评估指标
correct_predictions = sum([1 for p, t in zip(predictions, true_labels) if p == t])
accuracy = correct_predictions / len(true_labels)

print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# 你也可以保存预测结果到文件
results_df = pd.DataFrame({
    'true_labels': true_labels,
    'predictions': predictions
})
results_df.to_csv('test_results.csv', index=False)
