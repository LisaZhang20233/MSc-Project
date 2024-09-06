import os

log_dir = './logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 检查是否生成了日志文件
log_files = os.listdir(log_dir)
print(f"Logs available in '{log_dir}': {log_files}")