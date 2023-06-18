import re

from cyrtranslit import to_cyrillic, to_latin
lat_to_cyr = str.maketrans("aekmhopctyx", "аекмнорстух")
cyr_to_lat = str.maketrans("аекмнорстух", "aekmhopctyx")
stop_words = set(['г', 'кг', 'шт', 'мл', 'л', 'литр', 'мг', 'гр', 'км', 'мм', 'mm', 'уп'])

def replace_camel_case(s):
    matches = len(re.findall(r'(?<=[a-zа-я])([A-ZА-Я])', s))
    if matches > 2:
        return re.sub(r'(?<=[a-zа-я])([A-ZА-Я])', r' \1', s)
    else:
        return s
    
def split_on_language_change(s):
    s = re.sub(r'(?<=[a-zа-я])(?=[A-ZА-Я0-9])', r' ', s)
    s = re.sub(r'(?<=[A-ZА-Я0-9])(?=[a-zа-я])', r' ', s)
    return s


def insert_space_after_one(s):
    return re.sub(r'(1)(?=[A-Za-zА-Яа-я])', r'\1 ', s)

def replace_zero(s):
    s = re.sub(r'(?<=[A-Za-z])0(?=[A-Za-z])', 'o', s)
    s = re.sub(r'(?<=[А-Яа-я])0(?=[А-Яа-я])', 'о', s)
    return s

def preprocess_text(text):
    
    text = replace_zero(text)
    
    text = re.sub('\d+', '1', text)  # replace numbers to 1
    text = replace_camel_case(text)
    text = re.sub('д/', 'для ', text)
    text = re.sub('Д/', 'для ', text)
    text = insert_space_after_one(text)
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuation
    words = []
    for w in text.lower().split():
#         if len(w) < 2:
#             continue
        
        num_eng_chars = len(re.findall(r'[a-z]', w))
        num_ru_chars = len(re.findall(r'[а-я]', w))
        if num_eng_chars and num_ru_chars:
            if num_eng_chars > num_ru_chars:
                w = w.translate(cyr_to_lat)
            else:
                w = w.translate(lat_to_cyr)
        
        if w in stop_words:
            continue
        
        # если нет транзиторов
        w = split_on_language_change(w)
        if ' ' in w:
            words.extend(w.split())
        else:
            if w == 'нести':
                w = 'nestea'
            if w == 'эпика':
                w = 'epica'
            if w == 'хелен':
                w = 'helen'
            if w == 'харпер':
                w = 'harper'
            if w == 'тимотей':
                w = 'timotei'
            if w == 'тесс':
                w = 'tess'
            if w == 'кроненбург':
                w = 'kronenbourg'
            if w == 'пай':
                w = 'pie'
            if w == 'чоко':
                w = 'choko'
            if w == 'салтон':
                w = 'salton'
            words.append(w)
        
    return words