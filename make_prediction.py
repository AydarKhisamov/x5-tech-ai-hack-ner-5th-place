import json

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from utils import get_tokens, id2tag, predictions_to_entities

model_checkpoint = './results/checkpoint-595'
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

# инициализация процессора для вычислений
if torch.backends.mps.is_built():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

with open('data/test.json', 'r') as file:
    test_data = json.load(file)

# сохранение предсказаний для тестовой выборки
predictions_output = []
for example in test_data:
    text = example['text']
    tokens = get_tokens(text)
    model_tokens = []
    input_ids = []
    for token in tokens:
        token_ids = tokenizer(token, add_special_tokens=False)['input_ids']
        input_ids += token_ids
        model_tokens.append(tokenizer.convert_ids_to_tokens(token_ids))
    attention_mask = [1 for i in range(len(input_ids))]

    # "знакомые" модели токены, по которым она будет делать предсказания
    inputs = {
        'input_ids': torch.tensor([input_ids], device=device),
        'attention_mask': torch.tensor([attention_mask], device=device),
    }

    # инференс
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

    # конвертация выходных значений модели в лейблы NER
    predicted_labels = [
        id2tag[p.item()]
        for p in predictions[0][:len(text)]
        if p.item() != -100
    ]

    # извлечение сущностей из текста на основе предсказаний
    entities = predictions_to_entities(
        text, tokens, model_tokens, predicted_labels,
    )

    predictions_output.append({'text': text, 'entities': entities})

# сохранение предсказаний в файле JSON с кодировкой UTF-8
output_file = 'data/submission.json'
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(predictions_output, file, ensure_ascii=False, indent=4)
