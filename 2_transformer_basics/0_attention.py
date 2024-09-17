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

# 3. 定义一个Embedding层
config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)

# 4. 将token ID映射到token embedding
inputs_embeds = token_emb(inputs.input_ids)

# 5. 构建Q, K, V, 计算注意力分数
Q = K = V = inputs_embeds
dim_k = K.shape[-1]
# 第二种写法: dim_k = K.size(-1)
scores = torch.bmm(Q, K.transpose(1, 2) / sqrt(dim_k))
# bmm是Batch Matrix-Matrix Multiplication

# 6. Softmax标准化注意力权重
weights = F.softmax(scores, dim=-1)
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
        # mask -- [batch_size,query_length, key_length]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float('inf'))

    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

# 8. 注意力头类
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        atten_outputs = scaled_dot_product_attention(self.q(query), self.k(key), self.v(value), query_mask, key_mask, mask)
        return atten_outputs

# 9. 多头Attention类
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        # 一共定义了12个AttentionHead
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        x = torch.cat([h(query, key, value, query_mask, key_mask, mask) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x

# 10. 将 BERT-base-uncased 模型输入验证是否工作正常
model_ckpt = "bert-base-uncased"
# (1) 首先定义一个tokenizer, 随后分词并返回pt类型的tensor
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "Go big or go home"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
# inputs是一个字典, print(inputs.keys()) = dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
config = AutoConfig.from_pretrained(model_ckpt)
# bert-base-uncased 的 hidden_size 是768
# (2) 利用config的词典和embed大小来定义一个Embedding
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
# bert-base-uncased 的 vocab_size 是30522
# (3) 把inputs转化为embeds
inputs_embeds = token_emb(inputs.input_ids)
# input_ids 是每个分词在词典里的id的tensor, [batch_size, seq_len]
# inputs_embeds 的大小是[batsh_size, seq_len, embed_dim]
# (4) 利用config初始化多头
multihead_attn = MultiHeadAttention(config)
query = key = value = inputs_embeds
# (5) 把嵌入后的序列当作输入输入到多头注意力模型内
atten_outputs = multihead_attn(query, key, value)


# 11. FeedForward层
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        # intermediate_size = 4 * hidden_size
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # p=0.1

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

feed_forward = FeedForward(config)
ff_output = feed_forward(atten_outputs)


# 12. 构建Transformer Encoder层:
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x, mask=None):
        hidden_state = self.layer_norm_1(x)
        x = x + self.attention(hidden_state, hidden_state, hidden_state, mask=mask)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

encoder_layer = TransformerEncoderLayer(config)
print(inputs_embeds.shape)
print(encoder_layer(inputs_embeds).size())


# 13. 构建词语和位置同时映射表示:
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)  # max_position_embeddings:  512
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        seq_length = input_ids.size(1)  # seq_len:  5
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)  # position id shape:  torch.Size([1, 5])
        token_embeddings = self.token_embeddings(input_ids)  # token_embeddings.shape:  torch.Size([1, 5, 768])
        position_embeddings = self.position_embeddings(position_ids)  # position_embeddings.shape:  torch.Size([1, 5, 768])
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

embedding_layer = Embeddings(config)
print(embedding_layer(inputs.input_ids).size())


# 14. 构建完整Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x, mask=None):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x

encoder = TransformerEncoder(config)
print(encoder(inputs.input_ids).size())


# 15. 构建Decoder里面的mask
seq_len = inputs.input_ids.size(-1)
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
print(mask[0])