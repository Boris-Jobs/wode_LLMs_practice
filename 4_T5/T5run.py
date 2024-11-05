import json
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from transformers import TrainerCallback

# 自定义回调函数，用于记录每个 epoch 的损失
class LossRecorder(TrainerCallback):
    def __init__(self):
        self.epoch_losses = []

    def on_epoch_end(self, args, state, control, **kwargs):
        # 在每个 epoch 结束时记录损失
        if state.log_history:
            epoch_loss = state.log_history[-1].get('loss', None)
            if epoch_loss is not None:
                self.epoch_losses.append(epoch_loss)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f) 
"""
[
    {"question": "一立方米水等于多少吨", "context": "...", "answer": "1吨"},
    {"question": "杭州心理咨询哪里最好", "context": "...", "answer": "阳光心理咨询中心"}
]
"""

def preprocess_data(data):
    inputs = [f"question: {item['question']} context: {item['context']}" for item in data]
    targets = [item['answer'] for item in data]
    return inputs, targets
"""
  inputs = [
      "question: 问题1 context: 上下文1",
      "question: 问题2 context: 上下文2",
      "question: 问题3 context: 上下文3",
      ...
  ]
  targets = [
      "答案1",
      "答案2",
      "答案3",
      ...
  ]
"""


data = load_data('../data/QA_T5/train.json')
inputs, targets = preprocess_data(data)

# 加载 T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')

def tokenize_data(examples):
    # 对输入文本进行tokenize
    model_inputs = tokenizer(examples['input_text'], max_length=512, truncation=True)
    # 对目标文本进行tokenize，并取出 'input_ids'
    labels = tokenizer(examples['target_text'], max_length=512, truncation=True)
    # 将 labels 转换为 list，以满足 Dataset.map 的要求
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


train_dataset = Dataset.from_dict({'input_text': inputs, 'target_text': targets})

# 使用 tokenize_data 进行处理
train_dataset = train_dataset.map(tokenize_data, batched=True)

# 加载 T5 模型
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# 配置训练参数
training_args = TrainingArguments(
    output_dir='./results',  # 模型检查点保存位置
    logging_dir='./logs',    # 日志文件保存位置
    logging_steps=10,        # 每10步记录一次日志
    save_strategy="no",      # 不自动保存检查点
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01
)

# 使用 DataCollatorForSeq2Seq 处理批处理中的填充问题
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 假设你有一个评估数据文件 'eval.json'
eval_data = load_data('../data/QA_T5/dev.json')  # 加载评估数据
eval_inputs, eval_targets = preprocess_data(eval_data)  # 处理评估数据

eval_dataset = Dataset.from_dict({'input_text': eval_inputs, 'target_text': eval_targets})
eval_dataset = eval_dataset.map(tokenize_data, batched=True)  # 对评估数据进行 tokenization

# 创建损失记录器
loss_recorder = LossRecorder()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,  # 使用 data_collator
    eval_dataset=eval_dataset,  # 添加 eval_dataset
    callbacks=[loss_recorder]  # 添加损失记录器回调
)

# 开始训练
trainer.train()

# 手动保存最终模型
trainer.save_model('./results/final_checkpoint')

model = T5ForConditionalGeneration.from_pretrained('./results/final_checkpoint')

# 绘制损失-epoch 曲线
def plot_loss(losses):
    if not losses:
        print("No losses recorded.")
        return
    plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.savefig('training_loss_per_epoch.png')  # 保存图形为文件
    plt.close()  # 关闭图形以释放内存

# 使用记录的损失绘制曲线
plot_loss(loss_recorder.epoch_losses)

# 生成答案
def generate_answer(context, question):
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    # 将 input_ids 移动到模型所在的设备
    input_ids = input_ids.to(model.device)
    outputs = model.generate(input_ids)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# 示例使用
context = "2015下半年教师资格证考试时间为11月1日，考生可于2015年10月26日—10月31日登录报名系统，根据提示下载pdf准考证文件。下载后，仔细核对个人信息，并直接打印成准考，按准考证上的要求到指定地点参加考试。"
question = "请用中文回答教师资格证面试准考证打印时间"
answer = generate_answer(context, question)
print(answer)

# 示例生成答案
def evaluate_bleu(model, tokenizer, dataset):
    bleu_scores = {'BLEU-1': [], 'BLEU-2': [], 'BLEU-3': [], 'BLEU-4': []}
    for example in dataset:
        input_text = example['input_text']
        input_ids = tokenizer(input_text, return_tensors='pt').input_ids
        # 将 input_ids 移动到模型所在的设备
        input_ids = input_ids.to(model.device)
        outputs = model.generate(input_ids)
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        reference = [example['target_text'].split()]
        candidate = generated_answer.split()

        bleu_scores['BLEU-1'].append(sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
        bleu_scores['BLEU-2'].append(sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
        bleu_scores['BLEU-3'].append(sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
        bleu_scores['BLEU-4'].append(sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))

    avg_bleu_scores = {key: sum(values) / len(values) for key, values in bleu_scores.items()}
    return avg_bleu_scores

# 使用示例
bleu_scores = evaluate_bleu(model, tokenizer, eval_dataset)
print("Average BLEU Scores:", bleu_scores)
