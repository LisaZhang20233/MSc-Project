from transformers import RobertaTokenizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

file_path = 'data_new_train.pkl'
data = pd.read_pickle(file_path)
'''
# Load the dataset
try:
    
    print(data['before_merge'].iloc[1])
    #print(data['traceback_type'].describe())
    print("Successfully loaded data with pandas.")
except Exception as e:
    print(f"Failed to load with pandas: {e}")
'''

# Initialize the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Encode the 'before_merge' column using the tokenizer
tokens = data['before_merge'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, padding='max_length', max_length=128))

# Encode the 'category' column into numerical labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['category'])

# Convert tokens and labels to lists for further processing
tokens_list = tokens.tolist()
labels_list = labels.tolist()

# Display a sample of the processed data
print(tokens_list[:2], labels_list[:2])
