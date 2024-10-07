import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# 加载数据集
dataset = load_dataset('json', data_files='../data/IDAT/train.json', cache_dir='./new_cache_dir')

# 加载预训练模型和tokenizer
model_name = "google-t5/t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 数据预处理
def preprocess_function(examples):
    inputs = [f"问题: {q}" for q in examples['question']]
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
def generate_answer(question):
    input_text = f"问题: {question}"
    input_ids = tokenizer.encode(question, return_tensors='pt')

    # 确保 input_ids 和模型在同一设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    model.to(device)

    # 设置 max_new_tokens 参数
    outputs = model.generate(input_ids, max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# 示例预测
question = "课程代号是什么"
print("答案：", generate_answer(question))