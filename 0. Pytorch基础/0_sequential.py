# -*- coding: utf-8 -*-
"""
Created on 2024-09-04 20:55:01

@author: borisσ, Chairman of FrameX Inc.

I hope to use AI or LLMs to help people better understand the world and humanity.

We are big fans of xAI.

I am recently interested in Multimodal LLMs.
"""


import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class WodeSeq(nn.Module):
    def __init__(self):
        super(WodeSeq, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

wodeseq = WodeSeq()
print(wodeseq)
input = torch.ones((64, 3, 32, 32))
output = wodeseq(input)
print(output.shape)

writer = SummaryWriter("sequential")
writer.add_graph(wodeseq, input)
writer.close()

