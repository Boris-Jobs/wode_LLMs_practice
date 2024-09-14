from torch.utils.data import Dataset, DataLoader, IterableDataset
import json, random, os, numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, BertPreTrainedModel, BertModel, AdamW, get_scheduler
from tqdm.auto import tqdm

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


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
    


# 2. 数据集非常巨大, 难以一次性加载到内存中, 继承IterableDataset构造
class IterableAFQMC(IterableDataset):
    def __init__(self, data_file):
        self.data_file = data_file

    def __iter__(self):
        with open(self.data_file, 'rt') as f:
            for line in f:
                sample = json.loads(line.strip())
                yield sample





# 3. 初始化Dataloader, 见if __name__ == '__main__':



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
    



# 6. 优化模型参数
def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')  # 数值右对齐, 字符长度为7
    finish_step_num = (epoch-1) * len(dataloader)  # len(dataloader)返回的是批次信息
    print("\nlen(dataloader) is: ", len(dataloader))
    print("len(dataloader.dataset) is: ", len(dataloader.dataset))

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
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)  # 进度条往前推进一个单位
    return total_loss

def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)  # len(dataloader)返回的是批次信息
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 累加预测正确的样本
            # (1) a.type(torch.float), (2) a.float()
    correct /= size
    print(f"{mode} Accuracy: {(100 * correct):>0.1f}%\n")  # 保留一位小数
    return correct





if __name__ == '__main__':


    # 0. 设置全局随机种子
    seed_everything(42)


    # 1. 初始化数据集
    train_data = AFQMC('../data/afqmc_public/train.json')  # Windows路径里的反斜杠\才需要r'x\x', 或 'xx\\xx'
    valid_data = AFQMC('../data/afqmc_public/dev.json')
    print(train_data[0])


    # 2. 初始化Iterable数据集
    train_iter_data = IterableAFQMC('../data/afqmc_public/train.json')
    print(next(iter(train_iter_data)))


    # 3. 构造Dataloader, 并完成分词与输入数据处理
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

    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fun)
    valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collate_fun)
    # ValueError: DataLoader with IterableDataset: expected unspecified shuffle option, but got shuffle=True
    # 以上错误: Iterable数据集不需要shuffle选项

    batch_X, batch_y = next(iter(train_dataloader))
    # [CLS] 是 101, [SEP] 是 102
    # batch_X shape:  {'input_ids': torch.Size([4, 30]), 'token_type_ids': torch.Size([4, 30]), 'attention_mask': torch.Size([4, 30])}
    # batch_y shape:  torch.Size([4])


    # 4. 利用BertPreTrainedModel类构造模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = AutoConfig.from_pretrained(checkpoint)
    model = BERTForPairwiseCLS.from_pretrained(checkpoint, config=config).to(device)
    batch_X = batch_X.to(device)
    outputs = model(batch_X)
    print(outputs.shape)


    # 5. 构造训练需要的optimizer, epochs, 总共训练次数, 学习率调度器, 损失函数
    optimizer = AdamW(model.parameters(), lr=1e-5, no_deprecation_warning=True)  # 学习率从5e-5线性降到0
    epochs = 3
    num_training_steps = epochs * len(train_dataloader)  # 用于学习率调度器
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    print("num_training_steps: ", num_training_steps)
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.
    best_acc = 0.
    for t in range(epochs):
        print(f"Epoch {t+1}/{epochs}\n------------------------------------")
        # 6. 开始训练与验证
        total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
        valid_acc = test_loop(valid_dataloader, model, 'Valid')
        if valid_acc > best_acc:
            best_acc = valid_acc
            print('saving new weights...\n')
            torch.save(model.state_dict(), f'epoch_{t+1}_valid_acc_{(100*valid_acc):0.1f}_model_weights.bin')
    print("Done!")