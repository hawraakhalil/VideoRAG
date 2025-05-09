from sentence_transformers import SentenceTransformer
import json
import os
import hashlib
import numpy as np

TRANSCRIPT_DIR = "data/transcripts/"
TEXT_EMBEDDINGS_DIR = "data/embeddings/text/"

def generate_hash_filename(url: str) -> str:
    return hashlib.md5(url.encode('utf-8')).hexdigest()

def load_transcript(url: str):
    hash_name = generate_hash_filename(url)
    path = os.path.join(TRANSCRIPT_DIR, f"{hash_name}_segments.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_text_embeddings(url: str, model_name="all-MiniLM-L6-v2"):
    os.makedirs(TEXT_EMBEDDINGS_DIR, exist_ok=True)
    print(f"Loading text embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    transcript = load_transcript(url)
    texts = [segment["text"] for segment in transcript]
    embeddings = model.encode(texts, show_progress_bar=True)

    output_path = os.path.join(TEXT_EMBEDDINGS_DIR, f"{generate_hash_filename(url)}.npy")
    np.save(output_path, embeddings)
    print(f"Saved text embeddings to {output_path}")

    return output_path

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=dARr3lGKwk8"
    generate_text_embeddings(youtube_url)
