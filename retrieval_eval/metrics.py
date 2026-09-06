from dataclasses import dataclass

@dataclass(frozen=True)
class EvalResult:
    recall_at_k: float
    mrr: float
    queries: int

def evaluate(retriever, queries, relevant, k=5):
    if len(queries) != len(relevant) or not queries or k < 1:
        raise ValueError("queries and labels must align and k must be positive")
    recall = 0.0; reciprocal = 0.0
    for q, gold in zip(queries, relevant):
        ranked = retriever.search(q, k); gold = set(gold)
        if not gold:
            raise ValueError("each query needs at least one relevant document")
        recall += len(set(ranked) & gold) / len(gold)
        for pos, idx in enumerate(ranked, 1):
            if idx in gold: reciprocal += 1 / pos; break
    return EvalResult(recall / len(queries), reciprocal / len(queries), len(queries))
