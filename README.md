# Retrieval Eval Lab

Dependency-free benchmark harness for testing whether a retriever returns evidence that actually supports a query. It includes a deterministic TF-IDF retriever, recall@k, MRR, and a per-query report suitable for CI regression checks.

## Run

```bash
python -m retrieval_eval --demo
python -m unittest discover -s tests -v
```

The corpus and relevance labels are explicit inputs, which keeps evaluation reproducible and prevents silently tuning against production traffic.

## Senior engineering notes

The explicit corpus, queries, and relevance labels make regressions reviewable in CI. Production RAG should add nDCG, groundedness checks, ACL filtering before retrieval, embedding/index versioning, and a reviewed golden set; recall alone does not prove answer correctness.
