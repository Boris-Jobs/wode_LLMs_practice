# -*- coding: utf-8 -*-
"""
Created on 2024-09-02 16:22:05

@author: borisσ, Chairman of FrameX Inc.

I hope to use AI or LLMs to help people better understand the world and humanity.

We are big fans of xAI.

I am recently interested in Multimodal LLMs.
"""

from torch import nn
import torch


class WodeNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)

    def forward(self, input):
        output = self.conv1(input)
        return output


if __name__ == "__main__":
    wodetest = WodeNet()
    wode_res = wodetest(torch.randn(1, 3, 3, 4))
    print(wode_res.shape)
