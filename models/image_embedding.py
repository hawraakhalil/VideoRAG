import os
import json
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import hashlib

KEYFRAME_DIR = "data/keyframes/"
IMAGE_EMBEDDINGS_DIR = "data/embeddings/image/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_hash_filename(url: str) -> str:
    """
    Generate a unique hash filename based on the URL.
    """
    return hashlib.md5(url.encode('utf-8')).hexdigest()

def generate_keyframe_directory(url: str) -> str:
    """
    Generate the directory path for keyframes based on the URL hash.
    """
    hash_name = generate_hash_filename(url)
    return os.path.join(KEYFRAME_DIR, hash_name)


def load_keyframe_metadata(url: str):
    """
    Loads keyframe metadata from the specific directory for the video URL.
    """
    keyframe_directory = generate_keyframe_directory(url)
    metadata_filename = os.path.join(keyframe_directory, "keyframes_metadata.json")
    
    if not os.path.exists(metadata_filename):
        raise FileNotFoundError(f"Metadata file not found at {metadata_filename}. Make sure keyframes are extracted first.")
    
    with open(metadata_filename, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_image_embeddings(url: str):
    """
    Generates image embeddings for keyframes extracted from the video corresponding to the URL.
    """
    os.makedirs(IMAGE_EMBEDDINGS_DIR, exist_ok=True)
    
    # Load the CLIP model and processor
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load the keyframe metadata
    metadata = load_keyframe_metadata(url)
    
    image_tensors = []
    image_paths = []

    # Load images based on the metadata
    keyframe_directory = generate_keyframe_directory(url)
    for frame in metadata:
        path = os.path.join(keyframe_directory, frame["filename"])
        try:
            image = Image.open(path).convert("RGB")
            image_tensors.append(image)
            image_paths.append(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    # Process images for embedding generation
    inputs = processor(images=image_tensors, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)

    # Convert to numpy array and save the embeddings
    embeddings = embeddings.cpu().numpy()
    output_path = os.path.join(IMAGE_EMBEDDINGS_DIR, f"{generate_hash_filename(url)}.npy")
    np.save(output_path, embeddings)
    print(f"Saved image embeddings to {output_path}")

    return output_path

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=dARr3lGKwk8"
    generate_image_embeddings(youtube_url)