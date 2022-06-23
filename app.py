from bson import encode
from flask import Flask, request
from index import index
from query import query_executor
from parse import file_parser

from sentence_transformers import SentenceTransformer

app = Flask(__name__)

PASSAGE_LENGTH_WORDS = 100

@app.route("/index", methods=['POST'])
def post_index():
    if request.headers["Content-Type"] != "text/plain":
        return 'Unsupported content type', 415

    indx = index.Index()
    parser = file_parser.FileParser()

    file_contents = request.data.decode("utf-8")
    # passages = parser.get_passages(file_contents)
    # passages = parser.get_rolling_passages(file_contents)
    passages = parser.get_passages_with_complete_sentences(file_contents)
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

def encode_passages(passages):
    # encode the passages
    model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3')
    return model.encode(passages)