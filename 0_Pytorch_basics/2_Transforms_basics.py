# -*- coding: utf-8 -*-
"""
Created on 2024-09-01 15:04:09

@author: borisσ, Chairman of FrameX Inc.
"""

from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    image_path = r"..\data\hymenoptera_data\train\ants\0013035.jpg"
    image = Image.open(image_path)

    # 首先设置好单个的transforms
    trans_totensor = transforms.ToTensor()
    trans_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    trans_resize = transforms.Resize((256, 256))
    trans_resize2 = transforms.Resize(128)
    # 随后设置Compose组合各种transforms
    transform = transforms.Compose([trans_resize, trans_totensor, trans_normalize])
    transform2 = transforms.Compose([trans_resize2, trans_totensor, trans_normalize])

    image_transformed = transform(image)
    image_transformed2 = transform2(image)

    # 写入tensorboard中:
    writer = SummaryWriter(log_dir="logs")
    image = trans_totensor(image)
    writer.add_image("image", image, 0)
    # 下面两条代码体会一下global_step的用法.
    writer.add_image("image_transformed", image_transformed, 0)
    writer.add_image("image_transformed", image_transformed2, 1)
    writer.close()
    # cmd: tensorboard --logdir=logs
