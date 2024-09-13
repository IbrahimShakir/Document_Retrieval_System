from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from models import db, Document, User
from utils import vectorize_query, search_documents
from redis import Redis
import numpy as np
import time

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///documents.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
redis_client = Redis(host='localhost', port=6379, db=0)  # Redis for caching

# Initialize rate limiting
limiter = Limiter(app, key_func=get_remote_address, default_limits=["5 per minute"])

# Endpoint to check if API is active
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "active"}), 200

# Endpoint to search documents
@app.route('/search', methods=['POST'])
@limiter.limit("5 per minute", key_func=lambda: request.json['user_id'])
def search():
    start_time = time.time()
    data = request.json
    query = data.get('text', '')
    top_k = data.get('top_k', 5)
    threshold = data.get('threshold', 0.5)
    user_id = data['user_id']

    # Check cache first
    cache_key = f"search:{query}:{top_k}:{threshold}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return jsonify({"results": cached_result, "cached": True}), 200

    # Fetch user from DB and update request count
    user = User.query.filter_by(id=user_id).first()
    if not user:
        user = User(id=user_id, request_count=1)
        db.session.add(user)
    else:
        user.request_count += 1
    db.session.commit()

    # Retrieve top-k documents based on similarity
    query_vector = vectorize_query(query)
    results = search_documents(query_vector, top_k, threshold)

    # Cache the results
    redis_client.set(cache_key, jsonify(results), ex=300)  # Cache expires in 5 minutes

    # Record and return the response
    inference_time = time.time() - start_time
    response = {"results": results, "inference_time": inference_time}
    return jsonify(response), 200

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
