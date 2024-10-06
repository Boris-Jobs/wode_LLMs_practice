import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# 加载数据集
dataset = load_dataset('json', data_files='../data/QA_T5/train.json', cache_dir='./new_cache_dir')

# 加载预训练模型和tokenizer
model_name = "google-t5/t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 数据预处理
def preprocess_function(examples):
    inputs = [f"问题: {q} 上下文: {c}" for q, c in zip(examples['question'], examples['context'])]
    targets = examples['answer']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)
print(tokenized_datasets.keys())

# 加载数据集后
print(f"加载的数据集样本数: {len(dataset['train'])}")

# 数据预处理后
print(f"预处理后的样本数: {len(tokenized_datasets['train'])}")

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train']
)

# 打印训练数据集样本数
print(f"训练数据集样本数: {len(trainer.train_dataset)}")

trainer.train()

# 预测函数
def generate_answer(context, question):
    input_text = f"问题: {question} 上下文: {context}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# 示例预测
context = "你的上下文文本"
question = "你的问题"
print(generate_answer(context, question))