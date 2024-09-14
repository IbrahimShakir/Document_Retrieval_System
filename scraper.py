import requests
import feedparser
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel, pipeline
import torch
from chromadb import Client

# Initialize ChromaDB client and BERT model
client = Client()
collection = client.get_or_create_collection("documents")

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load the NER pipeline for extracting named entities from queries
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")


def get_bert_embedding(text: str):
    """Generate BERT embeddings for a given text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings[0]


def extract_domain_from_query(query: str):
    """Extract the most probable domain (organization or location) from the user's query using NER."""
    entities = ner_pipeline(query)
    domain_candidates = [entity['word'] for entity in entities if entity['entity'] in ['B-ORG', 'B-LOC']]
    
    # Return the most probable domain
    return domain_candidates[0] if domain_candidates else None


def scrape_articles_from_rss(rss_url: str):
    """Scrape articles from a given RSS feed URL."""
    feed = feedparser.parse(rss_url)
    articles = []
    
    for entry in feed.entries:
        title = entry.title
        link = entry.link
        published = entry.published
        
        # Scrape the article content
        response = requests.get(link)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        
        articles.append({
            'title': title,
            'link': link,
            'published': published,
            'content': content
        })
    
    return articles


def insert_into_chromadb(document):
    """Insert article content and metadata into ChromaDB."""
    doc_id = document["link"]
    
    try:
        embeddings = get_bert_embedding(document["content"])
        collection.add(
            documents=[document["content"]],
            metadatas=[{
                "title": document["title"],
                "url": document["link"]
            }],
            ids=[doc_id],
            embeddings=[embeddings]
        )
        print(f"Inserted document: {document['title']}")
    except Exception as e:
        print(f"Error inserting document into ChromaDB: {e}")


def scrape_articles(query: str):
    """Main function to handle scraping based on user's query."""
    # Extract the domain (e.g., organization, location) from the user's query
    domain = extract_domain_from_query(query)
    
    if not domain:
        raise ValueError("Could not extract a valid domain from the query.")
    
    # Generate the RSS feed URL based on the extracted domain
    rss_url = f"https://news.google.com/rss/search?q={domain}&hl=en-US&gl=US&ceid=US:en"
    
    # Scrape articles from the RSS feed
    articles = scrape_articles_from_rss(rss_url)
    
    # Insert articles into ChromaDB
    for article in articles:
        insert_into_chromadb(article)
