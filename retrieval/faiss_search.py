import faiss
import numpy as np
import os
import hashlib
import torch
from sentence_transformers import SentenceTransformer

TEXT_EMBEDDINGS_DIR = "data/embeddings/text/"
IMAGE_EMBEDDINGS_DIR = "data/embeddings/image/"

def generate_hash_filename(url: str) -> str:
    """
    Generates a hash filename for a given URL using the MD5 hash algorithm.
    
    Args:
        url (str): The URL for which the hash filename needs to be generated.
    
    Returns:
        str: The MD5 hash of the URL in hexadecimal format.
    """
    return hashlib.md5(url.encode('utf-8')).hexdigest()

def load_embeddings(url: str, modality: str = "text"):
    """
    Loads pre-computed embeddings (either text or image) from the disk.
    
    Args:
        url (str): The URL associated with the embeddings to load.
        modality (str, default="text"): Specifies whether to load "text" or "image" embeddings.
    
    Returns:
        np.ndarray: The loaded embeddings as a NumPy array.
    
    Raises:
        FileNotFoundError: If the embeddings file does not exist at the expected location.
    """
    hash_name = generate_hash_filename(url)
    dir_path = TEXT_EMBEDDINGS_DIR if modality == "text" else IMAGE_EMBEDDINGS_DIR
    file_path = os.path.join(dir_path, f"{hash_name}.npy")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embeddings not found at {file_path}")
    
    return np.load(file_path)

def build_faiss_index(embeddings: np.ndarray):
    """
    Builds a FAISS index using the L2 (Euclidean) distance metric. This index allows for efficient similarity search.
    
    Args:
        embeddings (np.ndarray): The embeddings to be indexed. Expected to be a 2D NumPy array with shape (n_samples, embedding_dim).
    
    Returns:
        faiss.IndexFlatL2: A FAISS index that contains the embeddings and can be used for searching.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim) # FAISS index for L2 distance
    index.add(embeddings) # Adding embeddings to the index
    return index

def embed_query_text(query: str, model_name="all-MiniLM-L6-v2"):
    """
    Embeds the input query text into a vector using a pre-trained SentenceTransformer model.
    
    Args:
        query (str): The query text to be embedded.
        model_name (str, default="all-MiniLM-L6-v2"): The name of the SentenceTransformer model to use for embedding the query.
    
    Returns:
        np.ndarray: The embedded query as a NumPy array (1D).
    """
    try:
        # Try with default settings
        model = SentenceTransformer(model_name)
        return model.encode([query])
    except NotImplementedError:
        # Handle the meta tensor error by manually moving to CPU first
        print("Handling meta tensor error - using CPU explicitly")
        
        # Set device to CPU explicitly
        device = torch.device("cpu")
        
        # Load with device specified
        model = SentenceTransformer(model_name, device=device)
        
        # Ensure no gradient computation
        with torch.no_grad():
            return model.encode([query], convert_to_numpy=True)

def search_faiss(index, query_embedding: np.ndarray, top_k=3):
    """
    Performs a search on the FAISS index to find the most similar embeddings to the query.
    
    Args:
        index (faiss.Index): The FAISS index to search in.
        query_embedding (np.ndarray): The query embedding to search for.
        top_k (int, default=3): The number of top matches to return.
    
    Returns:
        tuple: A tuple containing two elements:
            - Indices of the top-k matches (np.ndarray).
            - Distances to the top-k matches (np.ndarray).
    """
    # Ensure query_embedding is the right shape and dtype
    if len(query_embedding.shape) == 2 and query_embedding.shape[0] == 1:
        query_embedding = query_embedding.astype(np.float32)
    else:
        # Reshape if necessary
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
    D, I = index.search(query_embedding, top_k)
    return I[0], D[0]  # I: indices, D: distances

if __name__ == "__main__":
    # Example usage
    url = "https://www.youtube.com/watch?v=dARr3lGKwk8"
    query = "Who is the narrator of the story?"

    # Load embeddings and build index
    text_embeddings = load_embeddings(url, modality="text")
    index = build_faiss_index(text_embeddings)

    # Embed query and search
    query_emb = embed_query_text(query)
    indices, distances = search_faiss(index, query_emb)

    print("Top matches (indices and distances):")
    for idx, dist in zip(indices, distances):
        print(f"Index {idx} with distance {dist:.4f}")
