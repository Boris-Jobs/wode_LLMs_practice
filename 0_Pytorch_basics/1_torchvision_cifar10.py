# -*- coding: utf-8 -*-
"""
Created on 2024-09-02 15:50:11

@author: borisσ, Chairman of FrameX Inc.

I hope to use AI or LLMs to help people better understand the world and humanity.

We are big fans of xAI.

I am recently interested in Multimodal LLMs.
"""

import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    test_data = torchvision.datasets.CIFAR10(
        root="../data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False
    )
    img, target = test_data[0]
    print(img.shape)
    print(target)

    # 在测试集中加载数据
    writer = SummaryWriter("Dataloader")
    step = 0
    for data in test_loader:
        imgs, targets = data
        print(imgs.shape)
        print(targets)
        writer.add_images("test_data", imgs, step)
        step += 1
    writer.close()
