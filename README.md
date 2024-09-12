# wode_LLMs_practice


每一小节的标题是Repository里对应的文件夹名称.








## 0_Pytorch_basics

视频网址: [https://www.bilibili.com/video/av74281036/](https://www.bilibili.com/video/av74281036/)

__p4-7, p10-13, p15-24, 32__

三个常用命令:
```python
import torch
# 1. dir()
dir(torch.nn.functional)
# 2. help()
help(torch.cuda.is_available)  
# 3. ??

# 注意不是help(torch.cuda.is_available())
# help()也可以等同于??
# 比如torch.cuda.is_available??
```

[多通道卷积示意图](https://zh-v2.d2l.ai/chapter_convolutional-neural-networks/channels.html): 

卷积核数量 = channel_input * channel_output

![](./pics/conv_example.png)


























## 2_transformer_basics


### token序列常规编码方式总结
RNN (如LSTM):  

$`\boldsymbol{y}_t =f(\boldsymbol{y}_{t-1},\boldsymbol{x}_t)  `$

CNN:  

$`\boldsymbol{y}_t = f(\boldsymbol{x}_{t-1},\boldsymbol{x}_t,\boldsymbol{x}_{t+1})`$

Attention:  

$`\boldsymbol{y}_t = f(\boldsymbol{x}_t,\boldsymbol{A},\boldsymbol{B})`$  

其中, A和B是另外的词语序列 (矩阵)


### 模型文件保存
.pth 是 PyTorch 社区中常用的文件扩展名，通常用于保存模型权重、模型状态或训练检查点

.bin 是一个通用的二进制文件扩展名，可能更常用于保存任意二进制数据

### 关于学习率, 1e-5和5e-5的差距非常大

蚂蚁金融语义相似度数据集

| 训练细节 | 正确率, epoch 1, 2, 3|
| :------:| :--------:| 
| 1e-5学习率, 训练shuffle=False | 69.4, 72.2, 73.7 |
| 5e-5学习率, 训练shuffle=False| 69.0, 69.0, 69.0 |
| 1e-5学习率, 训练shuffle=True | 72.6, 73.3, 73.9|
