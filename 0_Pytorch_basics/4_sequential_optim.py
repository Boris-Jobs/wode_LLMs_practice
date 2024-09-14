# -*- coding: utf-8 -*-
"""
Created on 2024-09-04 20:55:01

@author: borisσ, Chairman of FrameX Inc.

I hope to use AI or LLMs to help people better understand the world and humanity.

We are big fans of xAI.

I am recently interested in Multimodal LLMs.
"""

import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(
    "../data", train=False, transform=torchvision.transforms.ToTensor(), download=True
)

dataloader = DataLoader(dataset, batch_size=64, shuffle=False)


class Boris(nn.Module):
    def __init__(self):
        super(Boris, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
boris = Boris()
optim = torch.optim.SGD(boris.parameters(), lr=0.01)
for epoch in range(200):
    running_loss = 0.0
    for data in dataloader:
        images, targets = data
        outputs = boris(images)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)

torch.save(boris.state_dict(), "boris.pth")
