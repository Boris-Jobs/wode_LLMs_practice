import torch
from transformers import AutoTokenizer, AutoConfig, BertPreTrainedModel, BertModel
from torch import nn

class BERTForPairwiseCLS(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 2)
        self.post_init()

    def forward(self, x):
        bert_output = self.bert(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)  # 保留下来的值放大 1/(1 - p) 倍
        logits = self.classifier(cls_vectors)
        return logits

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 初始化 tokenizer 和模型
checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
config = AutoConfig.from_pretrained(checkpoint)
model = BERTForPairwiseCLS.from_pretrained(checkpoint, config=config).to(device)

# 加载微调模型权重
model.load_state_dict(torch.load('epoch_3_valid_acc_73.9_model_weights.bin'))
model.eval()

def predict(sentence1, sentence2):
    # 编码输入
    inputs = tokenizer(sentence1, sentence2, padding=True, truncation=True, return_tensors="pt").to(device)
    print(inputs)
    # 模型预测
    with torch.no_grad():
        logits = model(inputs)
    prediction = logits.argmax(dim=1).item()
    return prediction

if __name__ == "__main__":
    print("""
          请输入两句话进行测试。输入格式: 句子1 句子2
          "sentence1": "借呗打开显示申请额度是什么", "sentence2": "申请借呗后才可以知道额度吗"
          "sentence1": "提示可以花呗分期付款", "sentence2": "花呗分期付款，产生退货"
          """)
    while True:
        try:
            sentence1 = input("句子1: ")
            sentence2 = input("句子2: ")
            prediction = "No! No!." if predict(sentence1, sentence2)==0 else "Yes! Yes!."
            print(f"模型预测结果: {prediction}")
        except KeyboardInterrupt:
            print("\n测试结束。")
            break
