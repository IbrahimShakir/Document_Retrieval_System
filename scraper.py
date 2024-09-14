import requests
from bs4 import BeautifulSoup
from chromadb import Client
from transformers import BertTokenizer, BertModel
import torch
import time

# Initialize ChromaDB client and BERT model
client = Client()
collection = client.get_or_create_collection("documents")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text: str):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings[0]

def scrape_articles():
    while True:
        # Example: Scrape articles from a RSS feed or news website
        url = "https://example-news-site.com/rss"  # Replace with your target site
        response = requests.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "xml")
            for item in soup.find_all("item"):
                title = item.title.text
                link = item.link.text
                description = item.description.text
                article_text = fetch_article_content(link)
                document = {
                    "title": title,
                    "url": link,
                    "content": article_text or description
                }
                insert_into_chromadb(document)
        else:
            print(f"Failed to scrape {url}, status code: {response.status_code}")
        time.sleep(600)  # Sleep for 10 minutes

def fetch_article_content(url):
    try:
        article_response = requests.get(url)
        article_soup = BeautifulSoup(article_response.content, "html.parser")
        article_body = article_soup.find("div", class_="article-content")
        return article_body.get_text(separator="\n") if article_body else None
    except Exception as e:
        print(f"Error fetching article content from {url}: {e}")
        return None

def insert_into_chromadb(document):
    doc_id = document["url"]
    try:
        embeddings = get_bert_embedding(document["content"])
        collection.add(
            documents=[document["content"]],
            metadatas=[{
                "title": document["title"],
                "url": document["url"]
            }],
            ids=[doc_id],
            embeddings=[embeddings]
        )
        print(f"Inserted document: {document['title']}")
    except Exception as e:
        print(f"Error inserting document into ChromaDB: {e}")
