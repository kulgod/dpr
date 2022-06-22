import os
from pprint import pprint 
from sentence_transformers import SentenceTransformer

PASSAGE_LENGTH_WORDS = 100
TEST_FILE_REL_PATH = 'test_data.txt'

class Parser:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3')

    def get_passages(self, data): 
        words = data.split(' ') 
        all_passages = []
        n_words = len(words) 

        for i in range(0, n_words // PASSAGE_LENGTH_WORDS):
            starting_word_index = i * PASSAGE_LENGTH_WORDS
            passage_words = words[starting_word_index: starting_word_index + PASSAGE_LENGTH_WORDS]
            all_passages.append(' '.join(passage_words))

        return all_passages

    def encode(self, data):
        self.latest_passage_list = self.get_passages(data)
        return self.model.encode(self.latest_passage_list)


if __name__ == "__main__":
    test_parser = Parser()
    root_dir = os.path.dirname(__file__)
    
    encoding = ''

    with open(os.path.join(root_dir, TEST_FILE_REL_PATH)) as test_data:
        passages = test_data.read()
        encoding = test_parser.encode(passages)

    for passage in test_parser.latest_passage_list:
        print(len(passage.split(' ')))

    print(encoding)