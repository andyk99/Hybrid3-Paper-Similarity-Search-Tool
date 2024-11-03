# Hybrid3-Paper-Similarity-Search-Tool
This repository contains a Flask-based web application that helps researchers find relevant papers in the Hybrid Cubed Hybrid Perovskite database. Using an AI-driven approach, it enables similarity searches based on cosine similarity across vector embeddings for paper metadata components (title, abstract, keywords, authors) using SciBERT.

## Features
#### Text-Based Query Search: Input a topic or keywords to retrieve similar papers based on cosine similarity with existing entries in the Hybrid Cubed database.
#### Component-Wise Similarity Scoring: Provides a breakdown of similarity for each component (title, abstract, keywords, authors).
#### Overall Similarity Ranking: Papers are sorted by average similarity to the query for easy access to the most relevant results.

## Project Structure
#### app.py: Main Flask application script.
#### similarity_search.py & similarity_search2.py: Scripts to compute similarity scores using SciBERT embeddings and FAISS.
#### find_paper_metadata.py: Utility script to fetch and add metadata for papers from CrossRef using DOIs.
#### papers.json: JSON file containing sample metadata for papers in the Hybrid Cubed database.
