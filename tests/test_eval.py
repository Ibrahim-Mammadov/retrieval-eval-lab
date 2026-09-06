import unittest
from retrieval_eval import TfidfRetriever, evaluate

class EvalTests(unittest.TestCase):
    def test_metrics(self):
        r = TfidfRetriever(["cats sleep", "dogs run"])
        out = evaluate(r, ["cats", "dogs"], [[0], [1]], 1)
        self.assertEqual((out.recall_at_k, out.mrr), (1.0, 1.0))

if __name__ == "__main__": unittest.main()
