# 1. NLP基础

台大李宏毅视频网址: https://aistudio.baidu.com/education/group/info/2060

学习章节: 8. NLP任务总览; 9 & 10. BERT和它的家族; 11. GPT3

## NLP任务总览

主要任务: text-text, text-class

text-class: 文本分一个类, 每个字符一个类别

seq2seq + copy_mechanism

seq <SEP> seq 一起放入Model.

![NLP概述](../pics/overview_of_NLP.png "自然语言处理概述")

BERT是否需要word segmentation, parsing 以及 coreference resolution三种pre-processing呢?

i: seq  
o: class for each token  
(1) word segmentation, (2) parsing--涉及构建语法树, (3) coreference resolution, (4) extractive summarization

i: seq
o: seq
(1) abstractive summarization, (2) unsupervised machine translation, (3) grammar error correction

__unsupervised machine translation是最关键的, 没法弄到7000 * 7000的language pair来做一个翻译所有语言的大模型.__

i: seq  
o: cls  
(1) sentiment classification
