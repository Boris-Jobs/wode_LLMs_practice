from transformers import T5Tokenizer, T5ForConditionalGeneration
from T5run import preprocess_data, load_data, tokenize_data
# *******要是没给T5run添加if __name__ == '__main__': 这将直接导致T5run会在运行T5test的时候运行********
from datasets import Dataset
from nltk.translate.bleu_score import sentence_bleu

tokenizer = T5Tokenizer.from_pretrained('t5-base')

model = T5ForConditionalGeneration.from_pretrained('t5-base')

def generate_answer(context, question):
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    print('1. the input_ids is: ', input_ids)
    print('2. the result of tokenizer is: ', tokenizer(input_text, return_tensors='pt'))
    input_ids = input_ids.to(model.device)
    outputs = model.generate(input_ids)
    print('3. the outputs is: ', outputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

context = "人工智障爱酱,是A.I.Channel里的kizina ai,超可爱,B站上有搜up主AIchannel,他是专门搬运这个的,而且有字幕,不多说了我去舔爱酱了(泥垢kizinaai是A.I.Channel里的kizina ai,超可爱,B站上有搜up主AIchannel,他是专门搬运这个的,而且有字幕,不多说了我去舔爱酱了(泥垢|从左往右:1.《守护甜心》藤咲抚子2.《旋风管家》绫崎飒3.《玛利亚狂热》祇堂鞠也4.《萌菌物语》结城萤5.《笨蛋测验召唤兽》木下秀吉6.《黑执事》夏尔·凡多姆海威7.《南家三姐妹》真8.《黑塔利亚》王耀伪娘大集合"
question = "人工智障爱酱是什么"
answer = generate_answer(context, question)
print('the answer is: ', answer)

"""
{"context": "人工智障爱酱,是A.I.Channel里的kizina ai,超可爱,B站上有搜up主AIchannel,他是专门搬运这个的,而且有字幕,不多说了我去舔爱酱了(泥垢kizinaai是A.I.Channel里的kizina ai,超可爱,B站上有搜up主AIchannel,他是专门搬运这个的,而且有字幕,不多说了我去舔爱酱了(泥垢|从左往右:1.《守护甜心》藤咲抚子2.《旋风管家》绫崎飒3.《玛利亚狂热》祇堂鞠也4.《萌菌物语》结城萤5.《笨蛋测验召唤兽》木下秀吉6.《黑执事》夏尔·凡多姆海威7.《南家三姐妹》真8.《黑塔利亚》王耀伪娘大集合", "answer": "kizina ai", "question": "人工智障爱酱是什么", "id": 982}
"""

eval_data = load_data('../data/QA_T5/dev.json')
eval_inputs, eval_targets = preprocess_data(eval_data)
eval_dataset = Dataset.from_dict({'input_text': eval_inputs, 'target_text': eval_targets})
eval_dataset = eval_dataset.map(tokenize_data, batched=True)

def evaluate_bleu(model, tokenizer, dataset):
    bleu_scores = {'BLEU-1': [], 'BLEU-2': [], 'BLEU-3': [], 'BLEU-4': []}
    for example in dataset:
        input_text = example['input_text']
        input_ids = tokenizer(input_text, return_tensors='pt').input_ids
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

bleu_scores = evaluate_bleu(model, tokenizer, eval_dataset)
with open('result.txt', 'w') as f:
    f.write(f"Average BLEU Scores: {bleu_scores}")