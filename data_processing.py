import json

# 读取JSON文件
with open('data/IDAT/train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 将文本提取并写入passages文件
with open('data/IDAT/passages.txt', 'w', encoding='utf-8') as f:
    for entry in data:
        f.write(entry['text'] + '\n')