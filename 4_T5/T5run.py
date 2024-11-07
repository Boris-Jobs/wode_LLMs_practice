import json
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import TrainerCallback

tokenizer = T5Tokenizer.from_pretrained('t5-base')

class LossRecorder(TrainerCallback):
    def __init__(self):
        self.epoch_losses = []

    def on_epoch_end(self, args, state, control, **kwargs):
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

def tokenize_data(examples):
    model_inputs = tokenizer(examples['input_text'], max_length=512, truncation=True)
    labels = tokenizer(examples['target_text'], max_length=512, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

if __name__ == '__main__':
    data = load_data('../data/QA_T5/train.json')
    inputs, targets = preprocess_data(data)


    train_dataset = Dataset.from_dict({'input_text': inputs, 'target_text': targets})

    train_dataset = train_dataset.map(tokenize_data, batched=True)

    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    training_args = TrainingArguments(
        output_dir='./final_checkpoint',
        logging_dir='./logs',    # 日志文件保存位置
        logging_steps=100,        # 每10步记录一次日志
        save_strategy="no",      # 不自动保存检查点
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


    eval_data = load_data('../data/QA_T5/dev.json')
    eval_inputs, eval_targets = preprocess_data(eval_data)
    eval_dataset = Dataset.from_dict({'input_text': eval_inputs, 'target_text': eval_targets})
    eval_dataset = eval_dataset.map(tokenize_data, batched=True)  # 对评估数据进行 tokenization

    loss_recorder = LossRecorder()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
        callbacks=[loss_recorder]
    )

    trainer.train()

    trainer.save_model('./final_checkpoint')
