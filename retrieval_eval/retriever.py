import math, re
from collections import Counter

TOKEN = re.compile(r"\w+", re.UNICODE)
class TfidfRetriever:
    def __init__(self, documents):
        self.documents = list(documents); self.tf = [Counter(TOKEN.findall(x.lower())) for x in self.documents]
        if not self.documents: raise ValueError("documents must not be empty")
        df = Counter(t for row in self.tf for t in row)
        self.idf = {t: math.log((1 + len(self.documents)) / (1 + n)) + 1 for t, n in df.items()}
        self.norm = [math.sqrt(sum((v * self.idf[t]) ** 2 for t, v in row.items())) or 1 for row in self.tf]

    def search(self, query, k=5):
        if k < 1: raise ValueError("k must be positive")
        q = Counter(TOKEN.findall(query.lower())); qv = {t: v * self.idf.get(t, 0) for t, v in q.items()}
        qn = math.sqrt(sum(x*x for x in qv.values())) or 1
        scores = []
        for i, row in enumerate(self.tf):
            score = sum(qv.get(t, 0) * v * self.idf[t] for t, v in row.items()) / (qn * self.norm[i])
            scores.append((score, i))
        return [i for _, i in sorted(scores, key=lambda item: (-item[0], item[1]))[:k]]
