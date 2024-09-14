# -*- coding: utf-8 -*-
"""
Created on 2024-09-01 12:30:05

@author: borisσ, Chairman of FrameX Inc.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)  # 返回含有所有文件名称的list

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path, img_name)  # 返回当前索引为idx的文件的路径
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


if __name__ == "__main__":
    # 读取数据集Dataset
    train_root_dir = r"..\data\hymenoptera_data\train"
    ants_label_dir = "ants"
    bees_label_dir = "bees"
    ants_dataset = MyDataset(train_root_dir, ants_label_dir)
    bees_dataset = MyDataset(train_root_dir, bees_label_dir)
    # 拼接数据集
    datasets = ants_dataset + bees_dataset
    # 读取数据集实例
    eg_img, eg_label = datasets[5]  # dataset[0]本身是一个tuple
    # eg_img.show()
    # print(len(datasets), len(ants_dataset), len(bees_dataset))

    # 一种批量把图片label写入.txt的方法
    ants_full_label_dir = train_root_dir + r"\ants_labels"
    os.makedirs(ants_full_label_dir, exist_ok=True)
    for i in os.listdir(train_root_dir + r"\ants"):
        file_name = i.split(".")[0]
        with open(os.path.join(ants_full_label_dir, file_name + ".txt"), "w") as f:
            f.write("ants")
