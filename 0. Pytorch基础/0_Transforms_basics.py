# -*- coding: utf-8 -*-
"""
Created on 2024-09-01 15:04:09

@author: borisσ, Chairman of FrameX Inc.
"""

from torchvision import transforms
from PIL import Image


if __name__ == '__main__':
    image_path = r'C:\Users\borisσ\Desktop\boris_Inc\_Github\wode_LLMs_practice\hymenoptera_data\train\ants\0013035.jpg'
    image = Image.open(image_path)

    # 1. transforms如何使用?
    transform = transforms.Compose([transforms.ToTensor()])
    image_transformed = transform(image)
    print("The size of original image: ", image.size, "The shape of transformed image: ", image_transformed.shape)

    # 2.
