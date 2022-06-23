import this
import numpy as np
from sklearn.metrics import pairwise

class QueryExecutor:
    def __init__(self, index_hash: dict, passages: dict) -> this:
        # read index into matrix format
        self.index_hash = index_hash
        self.passages = passages

    def search(self, encoded_query):
        x = np.array(encoded_query).reshape(1,-1) # reshape to 1 x m
        y = np.array(list(self.index_hash.values())) # n x m

        scores = pairwise.cosine_similarity(x, y) # 1 x n

        top_result_index = np.argmax(scores, axis=1)[0]

        passage_ids = list(self.index_hash.keys())
        top_passage_id = passage_ids[top_result_index]

        return self.passages[top_passage_id]


# contains some basic testing logic
if __name__ == "__main__":
    test_index = {
        "sha1": [0, 0.1, 1],
        "sha2": [1, 1.1, 2],
    }
    passages = {
        "sha1": "Lorem ipsum",
        "sha2": "foo bar",
    }

    query = [0.5, 1, 2]
    print(np.array(list(test_index.values())))

    s = QueryExecutor(test_index, passages)
    print(s.search(query))