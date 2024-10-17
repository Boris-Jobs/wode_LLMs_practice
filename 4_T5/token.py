import os
from huggingface_hub import login

# 替换下面的字符串为您实际的访问令牌
access_token = "YOUR_ACCESS_TOKEN"

# 设置环境变量
os.environ["HF_API_TOKEN"] = access_token

# 或者使用huggingface_hub库来登录
login(token=access_token)