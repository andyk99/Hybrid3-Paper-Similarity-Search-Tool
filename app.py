from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from similarity_search import search_similar_papers

app = Flask(__name__)
with open("papers.json", "r") as f:
    papers = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

# Handle the query
@app.route('/search', methods=['POST'])
def search():
    query = request.data.decode("utf-8")  # Read the query as a plain string
    results = search_similar_papers(query)  # Call the search function
    for result in results:
        for key, value in result.items():
            if isinstance(value, np.float32):
                result[key] = float(value)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)