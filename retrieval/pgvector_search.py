import os
import numpy as np
import hashlib
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch

# Load environment variables for database connection
load_dotenv()

# Default connection parameters (can be overridden by environment variables)
DB_NAME = os.getenv("DB_NAME", "videorag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")


# Embedding directories (same as in faiss_search.py for consistency)
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
    Same as in faiss_search.py for consistency.
    
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

def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database.
    
    Returns:
        connection: A psycopg2 connection object.
    """
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        raise Exception(f"Error connecting to PostgreSQL database: {e}")

def init_pgvector_database(conn=None):
    """
    Initializes the database by:
    1. Creating the pgvector extension if not exists
    2. Creating tables for text and image embeddings if not exist
    3. Setting up IVFFLAT indexes on the embedding columns
    
    Args:
        conn: An optional existing database connection
    """
    conn_provided = conn is not None
    if not conn_provided:
        conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            # Enable the pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create tables for text embeddings with video_id, segment_id, embedding
            cur.execute("""
                CREATE TABLE IF NOT EXISTS text_embeddings (
                    id SERIAL PRIMARY KEY,
                    video_url TEXT NOT NULL,
                    segment_id INTEGER NOT NULL,
                    embedding vector(384) NOT NULL
                );
            """)
            
            # Create tables for image embeddings if needed
            cur.execute("""
                CREATE TABLE IF NOT EXISTS image_embeddings (
                    id SERIAL PRIMARY KEY,
                    video_url TEXT NOT NULL,
                    keyframe_id INTEGER NOT NULL,
                    embedding vector(384) NOT NULL
                );
            """)
            
            # Create IVFFLAT indexes if they don't exist
            # First check if the indexes exist
            cur.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE indexname = 'text_embeddings_ivfflat_idx';
            """)
            if cur.fetchone() is None:
                # Create IVFFLAT index for text embeddings
                # The parameters 100 and 4 are list size and probe depth which can be tuned
                cur.execute("""
                    CREATE INDEX text_embeddings_ivfflat_idx ON text_embeddings 
                    USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
                """)
            
            cur.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE indexname = 'image_embeddings_ivfflat_idx';
            """)
            if cur.fetchone() is None:
                # Create IVFFLAT index for image embeddings
                cur.execute("""
                    CREATE INDEX image_embeddings_ivfflat_idx ON image_embeddings 
                    USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
                """)
            
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise Exception(f"Error initializing pgvector database: {e}")
    finally:
        if not conn_provided:
            conn.close()

def store_embeddings_in_pgvector(url, embeddings, modality="text"):
    """
    Stores the embeddings in the PostgreSQL database with pgvector.
    
    Args:
        url (str): The video URL associated with the embeddings.
        embeddings (np.ndarray): The embeddings to store. Shape (n_samples, embedding_dim).
        modality (str): Either "text" or "image" to specify the embeddings type.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Ensure pgvector and tables are set up
            init_pgvector_database(conn)
            
            # Clear existing embeddings for this URL to avoid duplicates
            table_name = "text_embeddings" if modality == "text" else "image_embeddings"
            segment_col = "segment_id" if modality == "text" else "keyframe_id"
            
            cur.execute(f"DELETE FROM {table_name} WHERE video_url = %s", (url,))
            
            # Prepare data for bulk insert
            data = []
            for i in range(len(embeddings)):
                embedding_vector = embeddings[i].astype(float)  # Ensure proper type
                data.append((url, i, embedding_vector))
            
            # Insert the embeddings in bulk
            execute_values(
                cur,
                f"INSERT INTO {table_name} (video_url, {segment_col}, embedding) VALUES %s",
                data,
                template=f"(%s, %s, %s)"
            )
            
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise Exception(f"Error storing embeddings in pgvector: {e}")
    finally:
        conn.close()

def embed_query_text(query: str, model_name="all-MiniLM-L6-v2"):
    """
    Embeds the input query text into a vector using a pre-trained SentenceTransformer model.
    Same as in faiss_search.py for consistency.
    
    Args:
        query (str): The query text to be embedded.
        model_name (str, default="all-MiniLM-L6-v2"): The name of the SentenceTransformer model to use.
    
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

def search_pgvector(query_embedding: np.ndarray, url: str, modality="text", top_k=3):
    """
    Performs a similarity search in the PostgreSQL database with pgvector.
    
    Args:
        query_embedding (np.ndarray): The query embedding to search for.
        url (str): The video URL to search in.
        modality (str): Either "text" or "image" to specify which embeddings to search.
        top_k (int): The number of top results to return.
    
    Returns:
        tuple: A tuple containing two elements:
            - Indices of the top-k matches (list of int).
            - Distances to the top-k matches (list of float).
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Ensure query_embedding is properly formatted as a float array
            query_vector = query_embedding.reshape(-1).astype(float).tolist()
            
            # Select the appropriate table based on modality
            table_name = "text_embeddings" if modality == "text" else "image_embeddings"
            segment_col = "segment_id" if modality == "text" else "keyframe_id"
            
            # Perform the vector search using the IVFFLAT index
            query = f"""
                SELECT {segment_col}, embedding <-> %s::vector AS distance
                FROM {table_name}
                WHERE video_url = %s
                ORDER BY distance ASC
                LIMIT %s;
            """
            
            cur.execute(query, (query_vector, url, top_k))
            results = cur.fetchall()
            
            # Extract indices and distances
            indices = [result[0] for result in results]
            distances = [result[1] for result in results]
            
            return indices, distances
    except Exception as e:
        raise Exception(f"Error searching pgvector database: {e}")
    finally:
        conn.close()

def load_or_create_pgvector_index(url, modality="text"):
    """
    Loads the embeddings from disk and stores them in pgvector database if not already present.
    
    Args:
        url (str): The video URL.
        modality (str): Either "text" or "image".
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Check if embeddings for this URL already exist in the database
            table_name = "text_embeddings" if modality == "text" else "image_embeddings"
            cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE video_url = %s", (url,))
            count = cur.fetchone()[0]
            
            if count == 0:
                # Embeddings not in database, load from disk and store them
                embeddings = load_embeddings(url, modality)
                store_embeddings_in_pgvector(url, embeddings, modality)
                print(f"Loaded and stored {modality} embeddings for {url} in pgvector database")
            else:
                print(f"Using existing {modality} embeddings from pgvector database for {url}")
            
    except Exception as e:
        raise Exception(f"Error loading or creating pgvector index: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # Example usage
    url = "https://www.youtube.com/watch?v=dARr3lGKwk8"
    query = "Who is the narrator of the story?"

    # Initialize the database with pgvector extension
    init_pgvector_database()
    
    try:
        # Load embeddings if not already in database
        load_or_create_pgvector_index(url, modality="text")
        
        # Embed query and search
        query_emb = embed_query_text(query)
        indices, distances = search_pgvector(query_emb, url, top_k=3)
        
        print("Top matches (indices and distances):")
        for idx, dist in zip(indices, distances):
            print(f"Index {idx} with distance {dist:.4f}")
    except Exception as e:
        print(f"Error: {e}")
