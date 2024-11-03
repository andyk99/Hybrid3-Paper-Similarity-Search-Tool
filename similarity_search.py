import json
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

    # Load paper data from JSON
with open("papers.json", "r") as f:
    papers = json.load(f)

    # Initialize SciBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

    # Get embeddings
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    # Create and store embeddings for each paper
embeddings = []
for paper in papers:
    paper_embeddings = {
        "paper_id": paper["paper_id"],
        "title_embedding": get_embedding(paper["title"], tokenizer, model),
        "abstract_embedding": get_embedding(paper["abstract"], tokenizer, model),
        "keywords_embedding": get_embedding(" ".join(paper["keywords"]), tokenizer, model),
        "authors_embedding": get_embedding(" ".join(paper["authors"]), tokenizer, model)
    }
    embeddings.append(paper_embeddings)

    # Calculate component-wise similarities and return sorted results
def component_similarity(query, paper_embeddings):
    query_embedding = get_embedding(query, tokenizer, model)

        # Cos similarity assess vector angles, gives semantic similarity rather than magnitude difference 
    title_similarity = cosine_similarity([query_embedding], [paper_embeddings["title_embedding"]])[0][0]
    abstract_similarity = cosine_similarity([query_embedding], [paper_embeddings["abstract_embedding"]])[0][0]
    keywords_similarity = cosine_similarity([query_embedding], [paper_embeddings["keywords_embedding"]])[0][0]
    authors_similarity = cosine_similarity([query_embedding], [paper_embeddings["authors_embedding"]])[0][0]
    avg_similarity = (title_similarity + abstract_similarity + keywords_similarity + authors_similarity) / 4

    return {
        "paper_id": paper_embeddings["paper_id"],
        "title_similarity": title_similarity,
        "abstract_similarity": abstract_similarity,
        "keywords_similarity": keywords_similarity,
        "authors_similarity": authors_similarity,
        "average_similarity": avg_similarity
    }

def search_similar_papers(query):
    similarities = []
    for paper_embeddings in embeddings:
        similarity_scores = component_similarity(query, paper_embeddings)
        similarities.append(similarity_scores)

        # Sort papers by average similarity in descending order and limit to top 5
    sorted_similarities = sorted(similarities, key=lambda x: x["average_similarity"], reverse=True)[:5]

    return sorted_similarities
