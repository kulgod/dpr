from bson import encode
from flask import Flask, request
from index import index
from query import query_executor

from sentence_transformers import SentenceTransformer

app = Flask(__name__)

PASSAGE_LENGTH_WORDS = 100

@app.route("/index", methods=['POST'])
def post_index():
    if request.headers["Content-Type"] != "text/plain":
        return 'Unsupported content type', 415

    indx = index.Index()

    file_contents = request.data.decode("utf-8")
    passages = get_passages(file_contents)
    encodings = encode_passages(passages)

    identifiers = indx.add_to_index(passages, encodings)
    return str(identifiers)

@app.route("/query")
def get_query():
    if request.headers["Content-Type"] != "text/plain":
        return 'Unsupported content type', 415

    query_string = request.data.decode("utf-8")
    # TODO: validate length of query
    encoded_query = encode_passages([query_string])

    # Load the index from file and use them to query
    indx = index.Index()
    executor = query_executor.QueryExecutor(indx.index_map, indx.passages)

    return executor.search(encoded_query)

def get_passages(data): 
    words = data.split(' ') 
    all_passages = []
    n_words = len(words)

    for i in range(0, n_words // PASSAGE_LENGTH_WORDS):
        starting_word_index = i * PASSAGE_LENGTH_WORDS
        passage_words = words[starting_word_index: starting_word_index + PASSAGE_LENGTH_WORDS]
        all_passages.append(' '.join(passage_words))

    return all_passages

def encode_passages(passages):
    # encode the passages
    model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3')
    return model.encode(passages)