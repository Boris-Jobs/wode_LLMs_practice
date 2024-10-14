# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = "[PAD]", "[CLS]"  # padding符号, bert中综合信息符号


def build_dataset(config):
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, "r", encoding="UTF-8") as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, *labels = lin.split("\t")
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                topic_list = [
                    "操控",
                    "内饰",
                    "安全性",
                    "空间",
                    "舒适性",
                    "外观",
                    "动力",
                    "价格",
                    "配置",
                    "油耗",
                ]
                topic_labels = [0] * len(topic_list)
                emotion_score_val = 0
                for label in labels:
                    topic, score = label.split("#")
                    if topic in topic_list:
                        topic_labels[topic_list.index(topic)] = 1
                        emotion_score_val += int(score)  # 累加情感分数
                emotion_score = [1, 0, 0] if emotion_score_val < 0 else ([0, 1, 0] if emotion_score_val == 0 else [0, 0, 1])
                # 填充 token_ids 和 mask
                if len(token_ids) < pad_size:
                    token_ids += [0] * (pad_size - len(token_ids))  # 填充到 pad_size
                    mask = [1] * len(token_ids) + [0] * (
                        pad_size - len(token_ids)
                    )  # mask 也要填充
                else:
                    token_ids = token_ids[:pad_size]  # 截断到 pad_size
                    mask = [1] * pad_size  # mask 长度固定为 pad_size
                    seq_len = pad_size

                contents.append((token_ids, seq_len, mask, topic_labels, emotion_score))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)  # token_ids
        y = torch.LongTensor([_[3] for _ in datas]).to(self.device)  # topic_labels
        seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)  # seq_len
        mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)  # mask
        emotion_scores = torch.FloatTensor([_[4] for _ in datas]).to(self.device)  # emotion_scores

        return (x, seq_len, mask), y, emotion_scores  # 返回所有5个值

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size : len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[
                self.index * self.batch_size : (self.index + 1) * self.batch_size
            ]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
