# -*- coding: utf-8 -*-
"""
Created on 2024-09-05 14:09:32

@author: borisσ, Chairman of FrameX Inc.

I hope to use AI or LLMs to help people better understand the world and humanity.

We are big fans of xAI.

I am recently interested in Multimodal LLMs.
"""


import torch
import torchvision

from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, Flatten
from torch.utils.data import DataLoader

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()]
)


datasets = torchvision.datasets.CIFAR10(root="../data", train=True, transform=transform)
dataloader = DataLoader(datasets, batch_size=1)


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


model = Boris()
state_dict = torch.load("boris.pth")
model.load_state_dict(state_dict)
model.to(torch.device("cuda"))
print(model)

num_correct = 0
total_num = 2500
for i in range(total_num):
    image, label = datasets[i]
    image = image.to(torch.device("cuda"))

    image = torch.reshape(image, (1, 3, 32, 32))
    model.eval()
    with torch.no_grad():
        output = model(image)
    # print(output)
    # print("The label is: ", label, "The prediction is: ", output.argmax(1))
    if output.argmax(1) == label:
        num_correct += 1

print(num_correct, "The correct rate is: ", num_correct / total_num)
