# -*- coding: utf-8 -*-
"""
Created on 2024-09-01 15:04:09

@author: borisσ, Chairman of FrameX Inc.
"""

from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    image_path = r'C:\Users\borisσ\Desktop\boris_Inc\_Github\wode_LLMs_practice\hymenoptera_data\train\ants\0013035.jpg'
    image = Image.open(image_path)

    # 1. transforms如何使用?
    trans_totensor = transforms.ToTensor()
    trans_normalize = transforms.Normalize([6, 3, 2], [9, 3, 5])
    transform = transforms.Compose([trans_totensor, trans_normalize])
    image_transformed = transform(image)
    print("The size of original image: ", image.size, "The shape of transformed image: ", image_transformed.shape)
    # 写入tensorboard中:
    writer = SummaryWriter(log_dir='logs')
    writer.add_image('image_transformed', image_transformed)
    writer.close()
    print(image_transformed[2][0][0])
    # cmd: tensorboard --logdir=logs
