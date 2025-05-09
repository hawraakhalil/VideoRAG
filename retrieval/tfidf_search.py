import os
import pickle
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load or define paths
TFIDF_MODEL_PATH = "retrieval/tfidf_vectorizer.pkl"
TFIDF_DATA_PATH = "retrieval/tfidf_corpus.pkl"

def fit_tfidf_vectorizer(chunks):
    """
    Fit the TF-IDF vectorizer on the provided chunks and save the model.
    
    Args:
        chunks (list): List of transcript chunks with text fields
    """
    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer().fit(texts)
    vectors = vectorizer.transform(texts)

    # Save model & data
    os.makedirs(os.path.dirname(TFIDF_MODEL_PATH), exist_ok=True)
    with open(TFIDF_MODEL_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(TFIDF_DATA_PATH, "wb") as f:
        pickle.dump((texts, vectors, chunks), f)
    
    return vectorizer, vectors

def load_tfidf_model():
    """Load the TF-IDF model and corpus data."""
    if not os.path.exists(TFIDF_MODEL_PATH) or not os.path.exists(TFIDF_DATA_PATH):
        raise FileNotFoundError("TF-IDF model or corpus data not found")
    
    with open(TFIDF_MODEL_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(TFIDF_DATA_PATH, "rb") as f:
        texts, vectors, chunks = pickle.load(f)
    
    return vectorizer, texts, vectors, chunks

def tfidf_search(query, k=3):
    """
    Search for relevant content using TF-IDF and cosine similarity.
    
    Args:
        query (str): The query string
        k (int): Number of results to return
        
    Returns:
        list: List of relevant chunks
    """
    vectorizer, texts, vectors, chunks = load_tfidf_model()
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, vectors).flatten()
    top_indices = scores.argsort()[-k:][::-1]

    return [chunks[i] for i in top_indices if scores[i] > 0.1]

def embed_query_text(query):
    """
    Convert query string to its TF-IDF representation.
    This function is designed to be compatible with the retrieval factory.
    
    Args:
        query (str): Query string
        
    Returns:
        np.ndarray: Query embedding as a NumPy array
    """
    vectorizer, _, _, _ = load_tfidf_model()
    query_vec = vectorizer.transform([query]).toarray()[0]
    return query_vec

def load_embeddings(url, modality="text"):
    """
    Load embeddings for a specific video and modality.
    Compatible with RetrievalFactory pattern.
    
    Args:
        url (str): Video URL
        modality (str): "text" or "image"
        
    Returns:
        sparse matrix: The TF-IDF vectors
    """
    if modality != "text":
        raise ValueError(f"TF-IDF retrieval only supports 'text' modality, not '{modality}'")
    
    try:
        _, _, vectors, _ = load_tfidf_model()
        return vectors
    except FileNotFoundError:
        # If not found, try to build the TF-IDF vectors from segments file
        from models.whisper_transcriber import generate_segments_filename
        segments_path = generate_segments_filename(url)
        
        if not os.path.exists(segments_path):
            raise FileNotFoundError(f"No transcript file found for URL: {url}")
        
        with open(segments_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        
        vectorizer, vectors = fit_tfidf_vectorizer(segments)
        return vectors

def build_tfidf_index(vectors):
    """
    Placeholder function to match the API of other retrieval methods.
    For TF-IDF, the vectors themselves serve as the index.
    
    Args:
        vectors: The TF-IDF vectors
        
    Returns:
        Same vectors (no additional indexing needed)
    """
    return vectors

def search_tfidf(index, query_embedding, top_k=3):
    """
    Search TF-IDF vectors using cosine similarity.
    Compatible with RetrievalFactory pattern.
    
    Args:
        index: The TF-IDF vectors
        query_embedding (np.ndarray): Query embedding
        top_k (int): Number of results to return
        
    Returns:
        tuple: (indices, distances) of top matches
    """
    # For TF-IDF, index is just the document vectors matrix
    vectors = index
    
    # Ensure query_embedding is correctly shaped for comparison
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Calculate cosine similarity
    scores = cosine_similarity(query_embedding, vectors).flatten()
    
    # Get top k indices and scores
    top_indices = scores.argsort()[-top_k:][::-1]
    top_scores = scores[top_indices]
    
    # Convert similarity scores to distances (1 - similarity)
    distances = 1.0 - top_scores
    
    return top_indices, distances
