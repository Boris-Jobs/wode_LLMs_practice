# coding: UTF-8
import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer
from torchviz import make_dot


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = "bert"
        self.train_path = dataset + "/data/train.txt"  # 训练集
        self.dev_path = dataset + "/data/dev.txt"  # 验证集
        self.test_path = dataset + "/data/test.txt"  # 测试集

        self.save_path = (
            dataset + "/saved_dict/" + self.model_name + ".ckpt"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_epochs = 10  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = "./bert_pretrain"
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.topic_fc = nn.Linear(config.hidden_size, 10)
        self.emo_fc = nn.Linear(config.hidden_size, 3)


    def forward(self, x):
        context = x[0]
        mask = x[2]

        outputs = self.bert(context, attention_mask=mask)
        pooled = outputs.pooler_output
        topic_out = self.topic_fc(pooled)
        emo_out = self.emo_fc(pooled)
        # make_dot(out, params=dict(self.named_parameters())).render("bert_model", format="pdf")
        return topic_out, emo_out


""" 
out:
tensor([[-0.7792, -0.6448,  0.3070,  ..., -0.9044, -1.6603, -2.0873],
        [ 1.4774, -1.4770, -0.0297,  ..., -1.8690, -1.5324, -2.2297],
        [-0.6536, -0.3918, -0.1490,  ..., -0.7804, -1.5847, -1.8200],
        ...,
        [ 2.1553,  6.1949,  2.0223,  ..., -2.9144, -1.7750, -2.8861],
        [-0.5619,  8.2569,  0.0913,  ..., -0.8796, -0.6909, -1.0884],
        [ 0.2276,  7.0054, -0.1130,  ..., -1.7861, -1.5647, -0.1931]],
       device='cuda:0')
torch.Size([128, 10])

pooled:
`pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a classifier pretrained on top of the hidden state associated to the first character of the input (`CLS`) to train on the Next-Sentence task (see BERT's paper).


x and x[i].shape for i in range(len(x))
(tensor([[ 101, 3118, 2875,  ...,    0,    0,    0],
        [ 101,  674, 3175,  ...,    0,    0,    0],
        [ 101, 4281, 2399,  ...,    0,    0,    0],
        ...,
        [ 101, 1057, 7305,  ...,    0,    0,    0],
        [ 101, 7350, 2168,  ...,    0,    0,    0],
        [ 101, 3772, 2356,  ...,    0,    0,    0]], device='cuda:0'), tensor([21, 21, 17, 20, 19, 15, 19, 11, 20, 14, 19, 13, 18, 20, 12, 15, 18, 16,
        21, 23, 19, 15, 27, 23, 16, 24, 23, 24, 16, 20, 18, 21, 24, 22, 17, 16,
        17, 12, 23, 16, 20, 21, 24, 21, 22, 23, 14, 18, 19, 18, 18, 18, 24, 25,
        18, 21, 17, 13, 23, 18, 13, 17, 16, 16, 18, 15, 25, 15, 19, 22, 22, 18,
        20, 24, 18, 21, 20, 17, 23, 23, 20, 17, 20, 15, 17, 18, 19, 22, 25, 18,
        15, 17, 23, 20, 11, 17, 22, 19, 14, 22, 17, 18, 18, 18, 22, 16, 24, 17,
        22, 24, 19, 16, 21, 15, 22, 22, 21, 25, 17, 20, 19, 23, 21, 18, 17, 18,
        16, 20], device='cuda:0'), tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]], device='cuda:0'))
torch.Size([128, 32])
torch.Size([128])  # However, in many single-sentence tasks, this tensor is either all zeros or a single integer indicating the sentence length.
torch.Size([128, 32])
"""
