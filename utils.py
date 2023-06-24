index_to_tag = ["O", "B-GOOD", "I-GOOD", "B-BRAND", "I-BRAND", "PAD"]
tag_to_index = {tag: index for index, tag in enumerate(index_to_tag)}

def apply_bio_tagging(row):
    """
    По токенам чека и разметке (то есть выделенным товарам и брендам) строим BIO-теги
    """
    tokens = row["tokens"]
    good = row["good"].split(',')[0].split()
    brand = row["brand"].split(',')[0].split()
    tags = ['O'] * len(tokens)
    for i, token in enumerate(tokens):
        if len(good) > 0 and tokens[i:i + len(good)] == good:
            tags[i] = "B-GOOD"
            for j in range(i + 1, i + len(good)):
                tags[j] = "I-GOOD"
        if len(brand) > 0 and tokens[i:i + len(brand)] == brand:
            tags[i] = "B-BRAND"
            for j in range(i + 1, i + len(brand)):
                tags[j] = "I-BRAND"
    return tags





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