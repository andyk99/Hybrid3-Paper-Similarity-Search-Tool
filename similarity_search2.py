import json
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

    # Load paper data from json
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

    # Create and store embeddings for components
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

    # Calculate similarity for individual components (assess angle between vectors rather than magnitude difference- best for semantic similarity)
def component_similarity(query, paper_embeddings):
    query_embedding = get_embedding(query, tokenizer, model)

    title_similarity = cosine_similarity([query_embedding], [paper_embeddings["title_embedding"]])[0][0]
    abstract_similarity = cosine_similarity([query_embedding], [paper_embeddings["abstract_embedding"]])[0][0]
    keywords_similarity = cosine_similarity([query_embedding], [paper_embeddings["keywords_embedding"]])[0][0]
    authors_similarity = cosine_similarity([query_embedding], [paper_embeddings["authors_embedding"]])[0][0]

        # Calculate average similarity to rank
    avg_similarity = (title_similarity + abstract_similarity + keywords_similarity + authors_similarity) / 4

        # Return similarity scores and average in a dictionary
    return {
        "paper_id": paper_embeddings["paper_id"],
        "title_similarity": title_similarity,
        "abstract_similarity": abstract_similarity,
        "keywords_similarity": keywords_similarity,
        "authors_similarity": authors_similarity,
        "average_similarity": avg_similarity
    }

    # Function to perform the similarity search and display ordered results
def search(query):
    similarities = []
        # Calculate similarity for each paper
    for paper_embeddings in embeddings:
        similarity_scores = component_similarity(query, paper_embeddings)
        similarities.append(similarity_scores)

        # Sort by average similarity in descending order
    sorted_similarities = sorted(similarities, key=lambda x: x["average_similarity"], reverse=True)

        # Display top 5 results neatly
    print(f"Query: {query}\n")
    for i, similarity in enumerate(sorted_similarities[:5], start=1):
        print(f"Rank {i} - Paper ID: {similarity['paper_id']}")
        print(f"Title Similarity: {similarity['title_similarity']:.4f}")
        print(f"Keywords Similarity: {similarity['keywords_similarity']:.4f}")
        print(f"Abstract Similarity: {similarity['abstract_similarity']:.4f}")
        print(f"Authors Similarity: {similarity['authors_similarity']:.4f}")
        print(f"Overall Similarity: {similarity['average_similarity']:.4f}")
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    search("mixed-halide")