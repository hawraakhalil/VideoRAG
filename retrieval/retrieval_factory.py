"""
Retrieval Factory Module

This module provides a factory for various retrieval methods in the VideoRAG system.
It allows for easy switching between different retrieval methods.
"""

from enum import Enum
import numpy as np

class RetrievalMethod(Enum):
    """Enum representing the available retrieval methods."""
    FAISS_FLAT = "FAISS Flat (In-Memory)"
    PGVECTOR_IVFFLAT = "PostgreSQL pgvector with IVFFLAT Index"
    TFIDF = "TF-IDF with Cosine Similarity"
    BM25 = "BM25 Okapi (Lexical Retrieval)"
    # To be added later:
    # LEXICAL_SEARCH = "Lexical Search"
    # FUSION = "Hybrid Search (Fusion)"

class RetrievalFactory:
    """Factory for creating and using different retrieval methods."""
    
    @staticmethod
    def get_method_names():
        """Returns a list of available retrieval method names for UI display."""
        return [method.value for method in RetrievalMethod]
    
    @staticmethod
    def load_embeddings(url, modality="text", method=RetrievalMethod.FAISS_FLAT):
        """
        Load embeddings using the specified retrieval method.
        
        Args:
            url (str): The video URL.
            modality (str): "text" or "image".
            method (RetrievalMethod): The retrieval method to use.
            
        Returns:
            The loaded embeddings or index, depending on the method.
        """
        # For TF-IDF and BM25, we only support text
        if method == RetrievalMethod.TFIDF and modality != "text":
            raise ValueError(f"TF-IDF retrieval only supports 'text' modality, not '{modality}'")
        if method == RetrievalMethod.BM25 and modality != "text":
            raise ValueError(f"BM25 retrieval only supports 'text' modality, not '{modality}'")
        
        if method == RetrievalMethod.FAISS_FLAT:
            from retrieval.faiss_search import load_embeddings, build_faiss_index
            embeddings = load_embeddings(url, modality)
            index = build_faiss_index(embeddings)
            return index
        
        elif method == RetrievalMethod.PGVECTOR_IVFFLAT:
            from retrieval.pgvector_search import load_or_create_pgvector_index
            # This will load or create the pgvector index
            load_or_create_pgvector_index(url, modality)
            # No need to return an index, as pgvector stores it in the database
            return None
        
        elif method == RetrievalMethod.TFIDF:
            from retrieval.tfidf_search import load_embeddings, build_tfidf_index
            vectors = load_embeddings(url, modality)
            index = build_tfidf_index(vectors)
            return index
        
        elif method == RetrievalMethod.BM25:
            from retrieval.bm25_search import load_embeddings, build_bm25_index
            corpus = load_embeddings(url, modality)
            index = build_bm25_index(corpus)
            return index
        
        else:
            raise ValueError(f"Unsupported retrieval method: {method}")
    
    @staticmethod
    def search(query_embedding, url, modality="text", method=RetrievalMethod.FAISS_FLAT, top_k=3):
        """
        Search for similar embeddings using the specified retrieval method.
        
        Args:
            query_embedding (np.ndarray): The query embedding.
            url (str): The video URL.
            modality (str): "text" or "image".
            method (RetrievalMethod): The retrieval method to use.
            top_k (int): Number of top results to return.
            
        Returns:
            tuple: (indices, distances) of the top matches.
        """
        # For TF-IDF and BM25, we only support text
        if method == RetrievalMethod.TFIDF and modality != "text":
            raise ValueError(f"TF-IDF retrieval only supports 'text' modality, not '{modality}'")
        if method == RetrievalMethod.BM25 and modality != "text":
            raise ValueError(f"BM25 retrieval only supports 'text' modality, not '{modality}'")
        
        if method == RetrievalMethod.FAISS_FLAT:
            from retrieval.faiss_search import load_embeddings, build_faiss_index, search_faiss
            embeddings = load_embeddings(url, modality)
            index = build_faiss_index(embeddings)
            return search_faiss(index, query_embedding, top_k)
        
        elif method == RetrievalMethod.PGVECTOR_IVFFLAT:
            from retrieval.pgvector_search import search_pgvector
            return search_pgvector(query_embedding, url, modality, top_k)
        
        elif method == RetrievalMethod.TFIDF:
            from retrieval.tfidf_search import load_embeddings, build_tfidf_index, search_tfidf
            vectors = load_embeddings(url, modality)
            index = build_tfidf_index(vectors)
            return search_tfidf(index, query_embedding, top_k)
        
        elif method == RetrievalMethod.BM25:
            print(f"RetrievalFactory: Using BM25 search with query embedding type: {type(query_embedding)}")
            from retrieval.bm25_search import load_embeddings, build_bm25_index, search_bm25
            corpus = load_embeddings(url, modality)
            print(f"RetrievalFactory: Got tokenized corpus with {len(corpus)} documents")
            index = build_bm25_index(corpus)
            print(f"RetrievalFactory: Built BM25 index, now searching...")
            results = search_bm25(index, query_embedding, top_k)
            print(f"RetrievalFactory: BM25 search returned {len(results[0])} results")
            return results
        
        else:
            raise ValueError(f"Unsupported retrieval method: {method}")
    
    @staticmethod
    def embed_query(query, method=RetrievalMethod.FAISS_FLAT):
        """
        Embed a query string into a vector.
        
        Args:
            query (str): The query string.
            method (RetrievalMethod): The retrieval method, which might use different embedding models.
            
        Returns:
            np.ndarray: The embedded query.
        """
        print(f"RetrievalFactory: Embedding query '{query}' using method {method.value}")
        
        # Different methods may use different embedding functions
        if method == RetrievalMethod.FAISS_FLAT:
            from retrieval.faiss_search import embed_query_text
            return embed_query_text(query)
        
        elif method == RetrievalMethod.PGVECTOR_IVFFLAT:
            from retrieval.pgvector_search import embed_query_text
            return embed_query_text(query)
        
        elif method == RetrievalMethod.TFIDF:
            from retrieval.tfidf_search import embed_query_text
            return embed_query_text(query)
        
        elif method == RetrievalMethod.BM25:
            from retrieval.bm25_search import embed_query_text
            tokens = embed_query_text(query)
            print(f"RetrievalFactory: BM25 tokenized query to {tokens}")
            return tokens
        
        else:
            raise ValueError(f"Unsupported retrieval method: {method}") 