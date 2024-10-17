import torch
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen-Base"  # 假设这是Qwen模型的标识符
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)

# Load dataset
dataset = load_dataset('json', data_files='../data/IDAT/train.json')

# Data preprocessing function
def preprocess_function(examples):
    inputs = examples['question']  # Expecting questions in English
    targets = examples['answer']  # Expecting answers in English

    # Tokenize inputs and targets, ensure consistent length
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length").input_ids

    # Mark padding tokens as -100 to ignore them in loss calculation
    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_sequence]
        for label_sequence in labels
    ]

    model_inputs["labels"] = labels
    return model_inputs

# Apply data preprocessing
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# # Set training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     num_train_epochs=20,
#     weight_decay=0.01,
# )

# # Create Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets['train'],
# )

# # Fine-tune the model
# trainer.train()

# Generate answer function
def generate_answer(question):
    # Encode input question
    input_ids = tokenizer.encode(question, return_tensors='pt')
    
    # Move model and input to GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    model.to(device)
    
    # Generate answer
    outputs = model.generate(input_ids, max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

while True:
    # 从终端获取用户输入作为问题
    question = input("Please enter your question: ")

    # 生成答案
    answer = generate_answer(question)

    # 打印答案
    print("Answer:", answer)