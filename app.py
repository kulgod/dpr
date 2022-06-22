from flask import Flask, request
from index import parser

app = Flask(__name__)

@app.route("/index", methods=['POST'])
def post_index():
    if request.headers["Content-Type"] != "text/plain":
        return 'Unsupported content type', 415

    p = parser.Parser()

    # parse file, pass to transformer.
    # 
    # note: this actually does both parsing & indexing. could be useful to 
    #   separate parsing out in the future & do more intense 
    file_contents = request.data.decode("utf-8")
    identifiers = p.encode(file_contents)

    return str(identifiers)

@app.route("/query")
def get_query():
    return "query"