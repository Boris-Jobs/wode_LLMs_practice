# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
from torchsummary import summary

# from transformers import AutoTokenizer

# # 下载分词器
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

# # 保存vocab文件到本地
# tokenizer.save_pretrained("./bert_pretrain")

parser = argparse.ArgumentParser(description="Chinese Text Classification")
parser.add_argument(
    "--model", type=str, required=True, help="choose a model: Bert, ERNIE"
)
args = parser.parse_args()
# 举例说明parser.parse_args()
# python run.py --model Bert --batch_size 64
# args = Namespace(model='Bert', batch_size=64)

if __name__ == "__main__":
    dataset = "caruser"

    x = import_module(args.model)  # 会导入models/bert.py
    config = x.Config(dataset)  # bert下的Config类
    np.random.seed(42)  # 设置np随机种子
    torch.manual_seed(42)  # 为当前CPU设置torch随机种子
    torch.cuda.manual_seed_all(42)  # 为所有GPU设置torch cuda随机种子
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    # summary(model, input_size=(32,), device='cuda')

    train(config, model, train_iter, dev_iter, test_iter)
