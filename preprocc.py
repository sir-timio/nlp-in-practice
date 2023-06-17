from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    NamesExtractor,

    Doc
)

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)

names_extractor = NamesExtractor(morph_vocab)

def lemmatize(words):
    text = ' '.join(words)
    
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    
    words = []
    for token in doc.tokens:
        words.append(token.lemma)
    return words

import re

from cyrtranslit import to_cyrillic

def preprocess_text(text):
    
    text = re.sub('\d+', '0', text)  # replace numbers to 0/''
    text = re.sub(r'(?<=[a-zа-я])([A-ZА-Я])', r' \1', text) # сплитим по заглав. букв
    text = re.sub('д/', 'для ', text)
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuation
    words = []
    for w in text.lower().split():
        if bool(re.search('[a-z]', w)) and bool(re.search('[а-я]', w)):
            words.append(to_cyrillic(w))
        else:
            words.append(w)
#     words = lemmatize(words)
    return words