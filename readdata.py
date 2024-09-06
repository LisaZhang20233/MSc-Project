import pickle
import pandas as pd

file_path = 'D:/PyCharm Community Edition 2022.1.4/MscProject/dataset/pytracebugs_dataset_v1/buggy_dataset/bugfixes_train.pickle'


# 尝试读取pickle文件
try:
    data = pd.read_pickle(file_path)
    print(data['before_merge'].iloc[1])
    #print(data['traceback_type'].describe())
    print("Successfully loaded data with pandas.")
except Exception as e:
    print(f"Failed to load with pandas: {e}")