import os
import pickle
import hashlib

from sentence_transformers import SentenceTransformer

PASSAGE_LENGTH_WORDS = 100
INDEX_PICKLE_FILE_PATH = 'index.pickle'
PASSAGE_IDENTIFIERS_PICKLE_FILE_PATH = 'passages.pickle'
TEST_FILE_REL_PATH = 'test_data.txt'

class Parser:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3')
        root_dir = os.path.dirname(__file__)

        self.index_file_path = os.path.join(root_dir, INDEX_PICKLE_FILE_PATH)
        self.index = {}

    # generates an encoding for each passage 
    def encode(self, data):
        # split data into passages 
        passages = self.get_passages(data)
        self.latest_passage_list = passages # cache passages for testing purposes

        # encode the passages
        encodings = self.model.encode(passages)

        # map each passage to an encoding using a generated identifier
        ids = [self.get_passage_id(p) for p in passages]
        passage_encodings = dict(zip(ids, encodings))
        passage_identifiers = dict(zip(ids, passages))

        # save encodings and passages to disk
        self.write_encodings_to_index(passage_encodings)
        self.write_passages(passage_identifiers)
        
        # return list of identifiers 
        return list(passage_encodings.keys())

    def get_passages(self, data): 
        words = data.split(' ') 
        all_passages = []
        n_words = len(words) 

        for i in range(0, n_words // PASSAGE_LENGTH_WORDS):
            starting_word_index = i * PASSAGE_LENGTH_WORDS
            passage_words = words[starting_word_index: starting_word_index + PASSAGE_LENGTH_WORDS]
            all_passages.append(' '.join(passage_words))

        return all_passages

    def get_passage_id(self, passage):
        return hashlib.sha256(passage.encode()).hexdigest()


    def write_encodings_to_index(self, passage_encodings):
        # ensure in-memory index is up to date
        self.update_index(passage_encodings)

        with open(self.index_file_path, 'wb') as f:
            pickle.dump(self.index, f)


    def update_index(self, passage_encodings):
        # if the index from disk contains data, load into memory & update with new state
        # otherwise, we can skip this and just add the new [id, encoding]
        if os.path.exists(self.index_file_path):
            with open(self.index_file_path, 'rb') as f:
                stored_index = pickle.load(f)
                if stored_index:
                    self.index = stored_index

        # merge existing index with new mappings
        self.index.update(passage_encodings)

    def write_passages(self, passage_identifiers):
        pass



# contains some basic testing logic
if __name__ == "__main__":
    test_parser = Parser()
    root_dir = os.path.dirname(__file__)
    
    # get list of encodings from parser
    encoding = []
    with open(os.path.join(test_parser.root_dir, TEST_FILE_REL_PATH)) as test_data:
        passages = test_data.read()
        encoding = test_parser.encode(passages)

    # check that the passages were split properly. 
    # length of each should be 100
    for passage in test_parser.latest_passage_list:
        print(len(passage.split(' ')))

    print(encoding)