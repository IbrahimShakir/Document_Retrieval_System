from fastapi import FastAPI, HTTPException
from fastapi.lifespan import Lifespan
from scraper import scrape_articles
import redis
import time
from threading import Thread
from transformers import BertTokenizer, BertModel
import torch
from chromadb import Client

app = FastAPI()

# Initialize ChromaDB client
client = Client()
collection = client.get_or_create_collection("documents")

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

cache = redis.Redis(host='localhost', port=6379, db=0)
user_db = {}

def get_bert_embedding(text: str):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings[0]

@app.on_event("lifespan")
async def lifespan(app: FastAPI):
    thread = Thread(target=scrape_articles)
    thread.start()
    yield
    thread.join()

@app.get("/health")
async def health():
    return {"status": "API is running"}

@app.get("/search")
async def search(text: str, top_k: int = 5, threshold: float = 0.8, user_id: str = None):
    start_time = time.time()
    
    if user_id in user_db and user_db[user_id] > 5:
        raise HTTPException(status_code=429, detail="Too many requests")
    
    if user_id in user_db:
        user_db[user_id] += 1
    else:
        user_db[user_id] = 1

    cached_result = cache.get(text)
    if cached_result:
        return {"results": cached_result.decode('utf-8')}

    query_embedding = get_bert_embedding(text)
    results = collection.query(
        embeddings=[query_embedding],
        n_results=top_k
    )
    
    cache.set(text, str(results), ex=3600)
    inference_time = time.time() - start_time
    return {"results": results, "inference_time": inference_time}
