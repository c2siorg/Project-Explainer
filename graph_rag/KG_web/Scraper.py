""" 
This file Scrapes the documentation from a given website including all sublinks using BeautifulSoup

Functions: 
scrape_page: Scrapes page to extract relevant details 
scrape_website: A function to recursively scrape all pages starting from the main documentation URL
build_index_from_documents: Builds an index from scraped documents
"""


import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

import time

def scrape_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.get_text(strip=True)
        links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
        return content, links
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None, []

def scrape_website(base_url, max_depth=3):
    visited = set()  
    to_visit = [base_url] 
    all_documents = [] 

    while to_visit:
        current_url = to_visit.pop(0)
        if current_url in visited or len(visited) >= max_depth:
            continue

        visited.add(current_url)

        content, links = scrape_page(current_url)
        if content:
            document = Document(
                text=content,
                metadata={'url': current_url}
            )
            all_documents.append(document)

            to_visit.extend(links)

        time.sleep(1)
    return all_documents

def build_index_from_documents(documents):
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = Ollama(model="llama3", request_timeout=360.0)

    index = VectorStoreIndex.from_documents(documents,)
    return index