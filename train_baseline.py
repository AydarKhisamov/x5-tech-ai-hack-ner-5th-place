import json
import random

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_metric
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments, set_seed)
from utils import get_labels, get_tokens, id2tag, tag2id

# установка параметра random_seed для воспроизводимости результатов
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
set_seed(seed)

with open('.../train.json', 'r') as f:
    train_data = json.load(f)

# словарь с датасетом
dataset_dict = {
    'id': [],
    'tokens': [],
    'ner_tags': [],
}

for idx, example in enumerate(train_data):
    tokens = get_tokens(example['text'])
    labels = get_labels(tokens, example['entities'])
    dataset_dict['id'].append(str(idx))
    dataset_dict['tokens'].append(tokens)
    dataset_dict['ner_tags'].append([tag2id[label] for label in labels])

full_dataset = Dataset.from_dict(dataset_dict)

# создание тренировочного и валидационного наборов данных
train_idx, val_idx = train_test_split(
    range(len(full_dataset)),
    test_size=0.15,
    random_state=seed,
    )

train_dataset = full_dataset.select(train_idx)
test_dataset = full_dataset.select(val_idx)

# создание DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
    })

# выбранная для обучения модель
model_checkpoint = 'google-bert/bert-base-multilingual-cased'
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(tag2id),
)

# инициализация процессора для вычислений
if torch.backends.mps.is_built():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
collator = DataCollatorForTokenClassification(
    tokenizer,
    padding='longest',
    max_length=512,
    )
metric = load_metric('seqeval')


# токенизация датасета
# конвертация выделенных из текста токенов в "знакомые" для модели токены
def tokenize_function(examples):
    """Конвертирует токен текста в токены модели и присваивает лейбл NER.

    Parameters:
        examples:
            Словарь с id текста, токенами текста и лейблами NER токенов.

    Returns:
        tokenized_inputs:
            Словарь со списками input_ids и labels для каждого текста.
    """
    tokenized_inputs = {'input_ids': [], 'labels': []}
    for tokens, tags in zip(examples['tokens'], examples['ner_tags']):
        input_ids = []
        labels = []
        for token, tag in zip(tokens, tags):
            token_ids = tokenizer(token, add_special_tokens=False)['input_ids']
            token_label = [tag for i in range(len(token_ids))]
            input_ids += token_ids
            labels += token_label
        tokenized_inputs['input_ids'].append(input_ids)
        tokenized_inputs['labels'].append(labels)
    return tokenized_inputs


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['tokens', 'ner_tags', 'id'],
)


def compute_metrics(p):
    """Расчитывает выбранную метрику точности модели.

    Parameters:
        p:
            Кортеж выходных значений модели и истинных лейблов NER.

    Returns:
        Словарь с метрикой и её значением.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2tag[p] for (p, l) in zip(prediction, label)
         if l not in {-100, tag2id['O']}]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [id2tag[l] for (p, l) in zip(prediction, label)
         if l not in {-100, tag2id['O']}]
        for prediction, label in zip(predictions, labels)
    ]

    score = metric.compute(
        predictions=true_predictions, references=true_labels,
    )

    return {'f1': score['overall_f1']}


training_args = TrainingArguments(
    output_dir='.../results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    save_strategy='epoch',
    save_steps=1,
    save_total_limit=1,
    report_to='none',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
