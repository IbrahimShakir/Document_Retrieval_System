import numpy as np
from models import Document

def vectorize_query(query):
    """
    Dummy function to convert query text to vector.
    Replace with a real embedding function like Sentence-BERT.
    """
    return np.random.rand(512)

def search_documents(query_vector, top_k, threshold):
    """
    Search documents by comparing query vector to document vectors.
    """
    documents = Document.query.all()
    results = []
    for doc in documents:
        similarity = np.dot(query_vector, np.array(doc.vector)) / (np.linalg.norm(query_vector) * np.linalg.norm(np.array(doc.vector)))
        if similarity >= threshold:
            results.append({"id": doc.id, "content": doc.content, "similarity": similarity})

    # Sort by similarity and return top_k results
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
    return results
