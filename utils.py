import random
import re

import numpy as np
import torch

index_to_tag = ["O", "B-GOOD", "I-GOOD", "B-BRAND", "I-BRAND", "PAD"]
tag_to_index = {tag: index for index, tag in enumerate(index_to_tag)}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_bio_tagging(row):
    """
    По токенам чека и разметке (то есть выделенным товарам и брендам) строим BIO-теги
    """
    tokens = row["tokens"]
    good = row["good"].split(",")[0].split()
    brand = row["brand"].split(",")[0].split()
    tags = ["O"] * len(tokens)
    for i, token in enumerate(tokens):
        if len(good) > 0 and tokens[i : i + len(good)] == good:
            tags[i] = "B-GOOD"
            for j in range(i + 1, i + len(good)):
                tags[j] = "I-GOOD"
        if len(brand) > 0 and tokens[i : i + len(brand)] == brand:
            tags[i] = "B-BRAND"
            for j in range(i + 1, i + len(brand)):
                tags[j] = "I-BRAND"
    return tags


def get_entities(label):
    label = label.replace("<\s>", "")
    good_regex = r"good:\s(.*?)(?:;|\s<|$)"
    brand_regex = r"brand:\s(.*?)(?:;|\s<|$)"
    good, brand = "", ""
    good_match = re.search(good_regex, label)
    brand_match = re.search(brand_regex, label)
    if good_match:
        good = good_match.group(1).strip()
    if brand_match:
        brand = brand_match.group(1).strip()
    return good, brand


def f1score(pred, target):
    pred = frozenset(x for x in pred)
    target = frozenset(x for x in target)
    tp = len(pred & target)
    fp = len(pred - target)
    fn = len(target - pred)
    if tp == 0:
        return 0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return 2 / (1 / precision + 1 / recall)


class F1Score:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, pred, target):
        pred = frozenset(x for x in pred)
        target = frozenset(x for x in target)
        self.tp += len(pred & target)
        self.fp += len(pred - target)
        self.fn += len(target - pred)

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def get(self):
        if self.tp == 0:
            return 0.0
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        return 2 / (1 / precision + 1 / recall)
