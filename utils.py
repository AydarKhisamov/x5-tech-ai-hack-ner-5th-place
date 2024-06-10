import re

import numpy as np

# уникальные теги NER
tags = [
    'TELEPHONE', 'ACRONYM', 'NAME', 'MAIL', 'LINK', 'ORG',
    'TECH', 'DATE', 'NUM', 'NAME_EMPLOYEE', 'PERCENT', 'O',
    ]

# присвоение id тегам NER
tag2id = {tag: id for id, tag in enumerate(tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

# паттерны некоторых сущностей
percent_pattern = r'\d+[.]\d+[%]'
telephone_pattern = r'[+]{,1}\d[\s(]{,2}\d{3}[\s)]{,2}\d{3}[-\s]{,1}\d{2}[-\s]{,1}\d{2}'
url_pattern = r'\bhttp[s]{,1}://w{,3}[.]{,1}\w*.\w{2,3}\b/{,1}'
date_pattern = r'\d{,2}.\d{2}.\d{4}'
mail_pattern = r'\b\w+@\w+[.]\w{2,4}\b'


def get_tokens(text):
    """Извлекает из текста токены.

    Parameters:
        text:
            Текст.

    Returns:
        Список токенов в изначальном порядке.
    """
    pos_dict = {}
    patterns = [
        # открывающий паттерн
        '[*]{1,}',
        # номер телефона
        telephone_pattern,
        # дата
        date_pattern,
        # адрес сайта
        url_pattern,
        # адрес эл.почты
        mail_pattern,
        # проценты
        percent_pattern,
        # слова
        r"\b[A-Za-zА-Яа-я]*[-'\w]*\w\b",
        # число
        r'\b\d{1,}\b',
        # закрывающие паттерны
        r'\b[^*]{1,}\b',
        '[^* ]{1,}',
        ]

    for pat in patterns:
        # поиск паттернов в тексте
        for match in re.finditer(pat, text):
            # сохранение токена с позицией в тексте
            pos_dict[match.start()] = match.group(0)

            # "заглушка" фрагмента для поиска по следующим паттернам
            text = re.sub(pat, '*' * len(match.group(0)), text, count=1)

    return [pos_dict[key] for key in sorted(pos_dict.keys())]


def get_labels(tokens, entities):
    """Присваивает каждому токену лейбл NER.

    Parameters:
        tokens:
            Токены из текста.
        entities:
            Список словарей с NER.

    Returns:
        labels:
            Список лейблов NER по кол-ву токенов.
    """
    labels = []
    for token in tokens:
        label = ['O']
        for entity in entities:
            if any([
                token == entity['word'],  # токен соответствует NER
                token in entity['word'].split(),  # токен внутри сложного NER
            ]):
                label = [entity['entity_group']]
                break
        labels += label

    return labels


def predictions_to_entities(text, text_tokens, model_tokens, predicted_labels):
    """На основании предсказаний модели присваивает токену лейбл NER.

    Parameters:
        text:
            Текст.
        text_tokens:
            Выделенные из текста токены.
        model_tokens:
            Токены, выделенные токенизатором модели.
        predicted_labels:
            Лейблы NER, предсказанные моделью для токенов от токенизатора.

    Returns:
        entities:
            Словарь с токеном, лейблом NER, посимвольным индексом токена в
            тексте.
    """
    previous_label = None
    entities = []
    t_counter = 0  # text counter
    l_counter = 0  # label counter
    for num, model_token in enumerate(model_tokens):
        # предсказанные лейблы для токенов из токенизатора,
        # которые соответствуют одному токену текста
        labels = predicted_labels[l_counter:l_counter+len(model_token)]

        # кол-во уникальных лейблов для одного токена текста
        unique_labels, counts = np.unique(labels, return_counts=True)

        # присвоение токену текста лейбла
        if len(unique_labels) > 0:

            # классификация по правилам
            if re.search(percent_pattern, text_tokens[num]):
                label = 'PERCENT'
            elif re.search(telephone_pattern, text_tokens[num]):
                label = 'TELEPHONE'
            elif re.search(url_pattern, text_tokens[num]):
                label = 'LINK'
            elif re.search(mail_pattern, text_tokens[num]):
                label = 'MAIL'
            elif re.search(date_pattern, text_tokens[num]):
                label = 'DATE'
            elif labels[0] == labels[-1]:
                label = labels[0]
            else:
                label = unique_labels[np.argsort(counts)][-1]

            if label != 'O':
                # поиск "координат" токена текста
                start = t_counter + text[t_counter:].find(text_tokens[num])
                end = start + len(text_tokens[num])
                t_counter = end
                if all([
                    previous_label in {'NAME', 'NAME_EMPLOYEE', 'ORG'},
                    label == previous_label,
                ]):
                    # конкатенация токена текста с предыдущим
                    entities[-1]['word'] += f' {text_tokens[num]}'
                    entities[-1]['end'] = end
                else:
                    # добавление токена текста в список NER
                    entity = {
                        'entity_group': label,
                        'word': text_tokens[num],
                        'start': start,
                        'end': end,
                        }
                    entities.append(entity)
            previous_label = label
        else:
            previous_label = None
        l_counter += len(model_token)

    # пост-процессинг
    for entity in entities:
        if entity['entity_group'] == 'ACRONYM':
            entity['start'] -= 1
            entity['end'] += 1
        if entity['entity_group'] == 'ORG':
            m = re.search(
                rf"{entity['word']} [(][-\w]+[)]", text[entity['start']:],
            )
            if m:
                entity['word'] = m.group(0)
                entity['end'] = entity['start'] + m.end()

    return entities
