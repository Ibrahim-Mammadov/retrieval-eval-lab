import argparse, json
from .retriever import TfidfRetriever
from .metrics import evaluate

def main():
    p = argparse.ArgumentParser(); p.add_argument("--demo", action="store_true"); a = p.parse_args()
    if not a.demo: p.error("use --demo")
    docs = ["Python reduces deployment friction", "GPU kernels accelerate tensor workloads", "Monitoring catches model drift"]
    result = evaluate(TfidfRetriever(docs), ["deployment", "tensor"], [[0], [1]], 2)
    print(json.dumps(result.__dict__))
if __name__ == "__main__": main()
