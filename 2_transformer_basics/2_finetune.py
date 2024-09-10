from torch.utils.data import Dataset, DataLoader, IterableDataset
import json
import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, BertPreTrainedModel, BertModel
from tqdm.auto import tqdm

# 1. 继承Dataset类构造自定义数据集
class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt') as f:  # 'rt' 指的是read text, 'r' 通常用于二进制文件如image, audio
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())  # strip()用来去除每一行开头和结尾的多余空白字符
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
train_data = AFQMC('./afqmc_public/train.json')  # Windows路径里的反斜杠\才需要r'x\x', 或 'xx\\xx'
valid_data = AFQMC('./afqmc_public/dev.json')

print(train_data[0])

# 2. 数据集非常巨大, 难以一次性加载到内存中, 继承IterableDataset构造
class IterableAFQMC(IterableDataset):
    def __init__(self, data_file):
        self.data_file = data_file

    def __iter__(self):
        with open(self.data_file, 'rt') as f:
            for line in f:
                sample = json.loads(line.strip())
                yield sample

train_data = IterableAFQMC('./afqmc_public/train.json')
print(next(iter(train_data)))


# 3. 构造Dataloader
checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def collate_fun(batch_samples):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1, 
        batch_sentence_2, 
        padding=True, 
        truncation=True, 
        return_tensors='pt'
    )
    y = torch.tensor(batch_label)
    return X, y

train_dataloader = DataLoader(train_data, batch_size=4, collate_fn=collate_fun)
# ValueError: DataLoader with IterableDataset: expected unspecified shuffle option, but got shuffle=True

batch_X, batch_y = next(iter(train_dataloader))
# [CLS] 是 101, [SEP] 是 102
# batch_X shape:  {'input_ids': torch.Size([4, 30]), 'token_type_ids': torch.Size([4, 30]), 'attention_mask': torch.Size([4, 30])}
# batch_y shape:  torch.Size([4])


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 4. 构建BERT模型
# C:\Users\borisσ\.cache\huggingface\hub
class BertForPairwiseCLS(nn.Module):
    def __init__(self):
        super(BertForPairwiseCLS, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)
        self.drop_out = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
    
    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        cls_vectors = self.drop_out(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits
    
model = BertForPairwiseCLS().to(device)
# print(model)


# 5. 用BertPreTrainedModel来初始化模型
class BERTForPairwiseCLS(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 2)
        self.post_init()

    def forward(self, x):
        bert_output = self.bert(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)  # 保留下来的值放大 1/(1 - p) 倍
        logits = self.classifier(cls_vectors)
        return logits
    
config = AutoConfig.from_pretrained(checkpoint)
model = BERTForPairwiseCLS.from_pretrained(checkpoint, config=config).to(device)
batch_X = batch_X.to(device)
outputs = model(batch_X)
print(outputs.shape)



# 6. 优化模型参数
def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = (epoch-1) * len(dataloader)

    model.train()
    for step, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        