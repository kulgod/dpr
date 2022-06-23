import os
import pickle
import hashlib

from sentence_transformers import SentenceTransformer

PASSAGE_LENGTH_WORDS = 100
INDEX_PICKLE_FILE_PATH = 'index.pickle'
PASSAGE_IDENTIFIERS_PICKLE_FILE_PATH = 'passages.pickle'
TEST_FILE_REL_PATH = 'test_data.txt'

class Index:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3')
        root_dir = os.path.dirname(__file__)

        self.index_file_path = os.path.join(root_dir, INDEX_PICKLE_FILE_PATH)
        self.passages_file_path = os.path.join(root_dir, PASSAGE_IDENTIFIERS_PICKLE_FILE_PATH)

        self.index_map = {}
        if os.path.exists(self.index_file_path):
            with open(self.index_file_path, 'rb') as f:
                stored_index = pickle.load(f)
                if stored_index:
                    self.index_map = stored_index

        self.passages = {}
        if os.path.exists(self.passages_file_path):
            with open(self.passages_file_path, 'rb') as f:
                stored_passages = pickle.load(f)
                if stored_passages:
                    self.passages = stored_passages


    def add_to_index(self, passages, encodings):
        # map each passage to an encoding using a generated identifier
        ids = [self.get_passage_id(p) for p in passages]
        passage_encodings = dict(zip(ids, encodings))
        passage_identifiers = dict(zip(ids, passages))

        # save encodings and passages to disk
        self.write_encodings_to_index(passage_encodings)
        self.write_passages(passage_identifiers)
        
        # return list of identifiers 
        return list(passage_encodings.keys())


    ## private class members ## 

    def get_passage_id(self, passage):
        return hashlib.sha256(passage.encode()).hexdigest()

    def write_encodings_to_index(self, passage_encodings):
        # merge existing index with new mappings and write to file
        self.index_map.update(passage_encodings)
        with open(self.index_file_path, 'wb') as f:
            pickle.dump(self.index_map, f)

    def write_passages(self, passage_identifiers):
        with open(self.passages_file_path, 'wb') as f:
            pickle.dump(passage_identifiers, f)
