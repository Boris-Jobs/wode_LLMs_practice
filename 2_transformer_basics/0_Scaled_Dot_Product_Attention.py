from torch import nn
from transformers import AutoConfig
from transformers import AutoTokenizer
from math import sqrt
import torch
import torch.nn.functional as F

# 1. 定义一个tokenizer
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# 2. 用tokenizer来进行分词
text = "Go big or go home."
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
# return_tensors="pt"说的是返回Pytorch张量
# 设置 add_special_tokens=False 去除了分词结果中的 [CLS] 和 [SEP]
print(inputs.input_ids)

# 3. 定义一个Embedding层
config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
print(token_emb)

# 4. 将token ID映射到token embedding
inputs_embeds = token_emb(inputs.input_ids)
print(inputs_embeds.size())

# 5. 构建Q, K, V, 计算注意力分数
Q = K = V = inputs_embeds
dim_k = K.shape[-1]
# 第二种写法: dim_k = K.size(-1)
scores = torch.bmm(Q, K.transpose(1, 2) / sqrt(dim_k))
print(scores.size())
# bmm是Batch Matrix-Matrix Multiplication

# 6. Softmax标准化注意力权重
weights = F.softmax(scores, dim=-1)
print(weights.sum(dim=-1))
atten_outputs = torch.bmm(weights, V)


# 7. 封装scaled dot product attention函数
def scaled_dot_product_attention(
    query, key, value, query_mask=None, key_mask=None, mask=None
):
    dim_k = query.shape[-1]
    scores = torch.bmm(query, key.transpose(1, 2))
    # [batch_size, query_length, key_length]
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
        # query_mask.unsqueeze(-1) -- [batch_size, query_length, 1]
        # key_mask.unsqueeze(1) -- [batch_size, 1, key_length]
        # mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float('inf'))

    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

# 8. 注意力头类
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().init()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        attn_outputs = scaled_dot_product_attention(self.q(query), self.k(key), self.v(value), query_mask, key_mask, mask)
        return atten_outputs

# 9. 多头Attention类
class MultiHeadAttention(nn.Module):
    def __init__(self, config):

        
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)