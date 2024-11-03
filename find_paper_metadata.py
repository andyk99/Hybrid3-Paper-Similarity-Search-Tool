import requests
import re
import argparse
import json
import os

def clean_doi(doi):
    """Cleans the DOI by removing any content before the '10.' prefix."""
    match = re.search(r'10\.\S+', doi)
    if match:
        return match.group(0)
    else:
        print("Invalid DOI format.")
        return None

def get_paper_metadata(doi, paper_id):
    """Fetches and appends paper metadata from CrossRef using the DOI."""
    cleaned_doi = clean_doi(doi)
    if not cleaned_doi:
        return None
    
    # Base URL for the CrossRef API
    base_url = "https://api.crossref.org/works/"
    url = base_url + cleaned_doi
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        item = data.get('message', {})
        
        # Extract metadata fields
        title = item.get('title', ['N/A'])[0]
        authors = item.get('author', [])
        authors_list = [f"{author.get('given', '')} {author.get('family', '')}".strip() for author in authors]
        abstract = item.get('abstract', 'N/A')
        keywords = item.get('subject', 'N/A')
        
        # Construct metadata dictionary with an id
        metadata = {
            'paper_id': paper_id,
            'title': title,
            'authors': authors_list,
            'abstract': abstract,
            'keywords': keywords
        }
        
        # Print the metadata for verification
        print(metadata)
        
        # Check if the file exists and load existing data
        if os.path.exists('papers.json'):
            # Load existing data
            with open('papers.json', 'r') as json_file:
                try:
                    existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    existing_data = []  # Start with an empty list if file is empty or corrupted
        else:
            existing_data = []  # Start with an empty list if the file doesn't exist
        
        # Append new metadata
        existing_data.append(metadata)
        
        # Save the updated list back to the JSON file
        with open('papers.json', 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)

        return metadata
    else:
        print(f"Failed to fetch data for DOI {doi}. Status code: {response.status_code}")
        return None

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Fetch paper metadata using a DOI and assign an ID.")
    parser.add_argument("doi", type=str, help="The DOI of the paper")
    parser.add_argument("id", type=int, help="The ID of the paper in the database")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Call the function with provided arguments
    get_paper_metadata(args.doi, args.id)
