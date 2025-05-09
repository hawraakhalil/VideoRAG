import os
import pickle
import json
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import hashlib

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load or define paths
BM25_MODEL_PATH = "retrieval/bm25_model.pkl"
BM25_DATA_PATH = "retrieval/bm25_corpus.pkl"

def tokenize(text):
    """Tokenize text using NLTK's word_tokenize."""
    return word_tokenize(text.lower())

def fit_bm25_index(chunks):
    """
    Fit the BM25 model on the provided chunks and save the model.
    
    Args:
        chunks (list): List of transcript chunks with text fields
        
    Returns:
        tuple: (bm25 model, tokenized corpus)
    """
    print(f"fit_bm25_index: Processing {len(chunks)} chunks")
    
    # Extract text from chunks
    corpus = [chunk["text"] for chunk in chunks]
    print(f"fit_bm25_index: Extracted corpus with {len(corpus)} documents")
    
    # Tokenize each document
    tokenized_corpus = []
    for i, doc in enumerate(corpus):
        tokens = tokenize(doc)
        # BM25 requires at least one token per document
        if not tokens:
            print(f"fit_bm25_index: Document {i} has no tokens, adding placeholder")
            tokens = ["empty_document"]
        tokenized_corpus.append(tokens)
    
    print(f"fit_bm25_index: Tokenized corpus has {len(tokenized_corpus)} documents")
    
    # Build BM25 model
    print("fit_bm25_index: Creating BM25Okapi model")
    bm25 = BM25Okapi(tokenized_corpus)
    print("fit_bm25_index: BM25 model created successfully")

    # Save model & data
    print(f"fit_bm25_index: Saving model to {BM25_MODEL_PATH}")
    os.makedirs(os.path.dirname(BM25_MODEL_PATH), exist_ok=True)
    
    with open(BM25_MODEL_PATH, "wb") as f:
        pickle.dump((bm25, tokenized_corpus), f)
    print(f"fit_bm25_index: Saved BM25 model with {len(tokenized_corpus)} documents")
    
    with open(BM25_DATA_PATH, "wb") as f:
        pickle.dump((corpus, chunks), f)
    print(f"fit_bm25_index: Saved corpus data with {len(corpus)} documents")
    
    # Verify we can reload the model
    try:
        with open(BM25_MODEL_PATH, "rb") as f:
            test_bm25, test_corpus = pickle.load(f)
        print(f"fit_bm25_index: Verified model can be loaded, contains {len(test_corpus)} documents")
    except Exception as e:
        print(f"fit_bm25_index: WARNING - Could not verify model: {str(e)}")
    
    return bm25, tokenized_corpus

def load_bm25_model():
    """Load the BM25 model and corpus data."""
    if not os.path.exists(BM25_MODEL_PATH):
        raise FileNotFoundError(f"BM25 model not found at {BM25_MODEL_PATH}")
    if not os.path.exists(BM25_DATA_PATH):
        raise FileNotFoundError(f"BM25 corpus data not found at {BM25_DATA_PATH}")
    
    try:
        with open(BM25_MODEL_PATH, "rb") as f:
            bm25, tokenized_corpus = pickle.load(f)
        
        with open(BM25_DATA_PATH, "rb") as f:
            corpus, chunks = pickle.load(f)
        
        print(f"Loaded BM25 model with {len(tokenized_corpus)} documents")
        return bm25, tokenized_corpus, corpus, chunks
    except Exception as e:
        # If there's an issue loading the pickled files, provide a helpful error
        raise RuntimeError(f"Error loading BM25 model: {str(e)}. Try regenerating the embeddings.")

def bm25_search(query, k=3):
    """
    Search for relevant content using BM25.
    
    Args:
        query (str): The query string
        k (int): Number of results to return
        
    Returns:
        list: List of relevant chunks
    """
    bm25, tokenized_corpus, corpus, chunks = load_bm25_model()
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [chunks[i] for i in top_indices if scores[i] > 0]

def embed_query_text(query):
    """
    Convert query string to tokenized form for BM25.
    This function is designed to be compatible with the retrieval factory.
    
    Args:
        query (str): Query string
        
    Returns:
        list: Tokenized query
    """
    print(f"BM25 embed_query_text: Processing query of type {type(query)}: '{query}'")
    
    # Make sure query is a string
    if not isinstance(query, str):
        try:
            query = str(query)
            print(f"BM25 embed_query_text: Converted query to string: '{query}'")
        except Exception as e:
            print(f"BM25 embed_query_text: Error converting query to string: {str(e)}")
            # Use an empty string as fallback
            query = ""
    
    # Tokenize and ensure we have tokens
    tokens = tokenize(query)
    print(f"BM25 embed_query_text: Tokenized to {len(tokens)} tokens: {tokens}")
    
    # BM25 requires at least one token to work
    if not tokens:
        print("BM25 embed_query_text: No tokens found, using placeholder")
        # Add a fallback token if tokenization resulted in empty list
        tokens = ["placeholder"]
    
    return tokens

def load_embeddings(url, modality="text"):
    """
    Load or create BM25 corpus and model.
    
    Args:
        url (str): The video URL (used for file path construction)
        modality (str): Only "text" is supported for BM25
        
    Returns:
        tokenized_corpus: The tokenized corpus for BM25
    """
    print(f"BM25: Loading embeddings for URL: {url}, modality: {modality}")
    if modality != "text":
        raise ValueError(f"BM25 retrieval only supports 'text' modality, not '{modality}'")
    
    # Create hash of URL for file naming
    url_hash = hashlib.md5(url.encode()).hexdigest()
    
    try:
        print(f"BM25: Attempting to load existing BM25 model")
        _, tokenized_corpus, _, _ = load_bm25_model()
        print(f"BM25: Successfully loaded existing model with {len(tokenized_corpus)} documents")
        return tokenized_corpus
    except FileNotFoundError as e:
        print(f"BM25: Model not found: {str(e)}")
        # If model doesn't exist, create it from segments file
        segment_path = f"data/transcripts/{url_hash}_segments.json"
        print(f"BM25: Creating new model from segments at: {segment_path}")
        
        if not os.path.exists(segment_path):
            print(f"BM25: ERROR - Segment file not found at: {segment_path}")
            raise FileNotFoundError(f"Cannot create BM25 model: Segment file not found at {segment_path}")
            
        with open(segment_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        
        print(f"BM25: Fitting BM25 index with {len(segments)} segments")
        _, tokenized_corpus = fit_bm25_index(segments)
        print(f"BM25: Returning tokenized corpus with {len(tokenized_corpus)} documents")
        return tokenized_corpus
    except Exception as e:
        print(f"BM25: Unexpected error loading embeddings: {str(e)}")
        raise

def build_bm25_index(tokenized_corpus):
    """
    Build or load BM25 index.
    
    Args:
        tokenized_corpus: The tokenized corpus
        
    Returns:
        BM25Okapi: The BM25 model
    """
    print(f"BM25: Building index from tokenized corpus with {len(tokenized_corpus)} documents")
    try:
        print("BM25: Attempting to load existing BM25 model")
        bm25, _, _, _ = load_bm25_model()
        print("BM25: Successfully loaded existing model")
        return bm25
    except FileNotFoundError:
        print("BM25: No existing model found, creating new BM25 model from tokenized corpus")
        # If we have a tokenized corpus but no saved model, create a new one
        if tokenized_corpus and len(tokenized_corpus) > 0:
            print(f"BM25: Creating new BM25 model with {len(tokenized_corpus)} documents")
            # Make sure all documents have at least one token
            for i, tokens in enumerate(tokenized_corpus):
                if not tokens:
                    print(f"BM25: Document {i} has no tokens, adding placeholder")
                    tokenized_corpus[i] = ["empty_document"]
            
            # Create a new BM25 model
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Save the model with a minimal corpus
            with open(BM25_MODEL_PATH, "wb") as f:
                pickle.dump((bm25, tokenized_corpus), f)
            with open(BM25_DATA_PATH, "wb") as f:
                # We need to create a minimal corpus and chunks data structure
                minimal_corpus = ["" for _ in range(len(tokenized_corpus))]
                minimal_chunks = [{"text": ""} for _ in range(len(tokenized_corpus))]
                pickle.dump((minimal_corpus, minimal_chunks), f)
            
            print("BM25: Created and saved new BM25 model")
            return bm25
        else:
            print("BM25: ERROR - Empty tokenized corpus provided")
            raise ValueError("Cannot create BM25 model: empty tokenized corpus")
    except Exception as e:
        print(f"BM25: Error in build_bm25_index: {str(e)}")
        raise

def search_bm25(index, query_embedding, top_k=3):
    """
    Search using BM25.
    Compatible with RetrievalFactory pattern.
    
    Args:
        index: The BM25 model
        query_embedding: Tokenized query (from embed_query_text)
        top_k (int): Number of results to return
        
    Returns:
        tuple: (indices, distances) of top matches
    """
    print(f"BM25: Searching with query tokens: {query_embedding}")
    # For BM25, index is the BM25 model and query_embedding is the tokenized query
    bm25 = index
    tokenized_query = query_embedding
    
    # Get BM25 scores
    print("BM25: Getting scores from BM25 model")
    scores = bm25.get_scores(tokenized_query)
    print(f"BM25: Got scores with shape: {scores.shape if hasattr(scores, 'shape') else len(scores)}")
    print(f"BM25: Score range: min={np.min(scores) if len(scores) > 0 else 'N/A'}, max={np.max(scores) if len(scores) > 0 else 'N/A'}")
    
    # Check if we have any meaningful scores
    if len(scores) == 0 or np.max(scores) <= 0:
        print("BM25: No meaningful scores found, returning empty results")
        # Return empty results if no matches
        return np.array([], dtype=int), np.array([], dtype=float)
    
    # Get top k indices and scores
    top_indices = np.argsort(scores)[-top_k:][::-1]
    top_scores = scores[top_indices]
    
    print(f"BM25: Top {len(top_indices)} indices: {top_indices}")
    print(f"BM25: Top scores: {top_scores}")
    
    # Convert BM25 scores to distances format expected by the retrieval factory
    # Higher BM25 score = better match, so we need to invert for distances
    # where lower distance = better match
    
    # BM25 scores can vary greatly depending on document length and query,
    # so we need to map them to a reasonable distance range (0-1)
    # that will result in meaningful confidence scores later
    
    # Scale distances to be between 0.1 and 0.9 
    # This will map to confidence scores between 0.1 and 0.9
    max_score = np.max(scores)
    if max_score > 0:
        # Normalize and invert: high scores -> low distances
        # Scale to 0.1-0.9 range for reasonable confidences
        normalized_scores = top_scores / max_score
        distances = 0.9 - (normalized_scores * 0.8)
    else:
        # If all scores are 0, use high distances
        distances = np.ones_like(top_scores) * 0.9
    
    print(f"BM25: Converted to distances: {distances}")
    return top_indices, distances
