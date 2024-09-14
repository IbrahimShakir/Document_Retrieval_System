from fastapi import FastAPI, HTTPException, Depends
import chromadb
import redis
import time
from threading import Thread
from fastapi.background import BackgroundTasks

app = FastAPI()
cache = redis.Redis(host='localhost', port=6379, db=0)
user_db = {}  # Simple in-memory user tracking

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection("documents")

def scrape_articles():
    while True:
        # Scrape news articles and insert into ChromaDB
        pass

@app.on_event("startup")
async def startup_event():
    # Start scraping thread
    thread = Thread(target=scrape_articles)
    thread.start()

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

    # Check cache
    cached_result = cache.get(text)
    if cached_result:
        return {"results": cached_result}

    # Query ChromaDB for top_k results
    results = collection.query(text, top_k=top_k, threshold=threshold)
    
    # Cache the result
    cache.set(text, results)

    # Calculate inference time
    inference_time = time.time() - start_time
    return {"results": results, "inference_time": inference_time}
