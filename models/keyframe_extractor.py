import cv2
import os
import json
import hashlib
from datetime import timedelta

VIDEO_DIR = "data/raw_video/"
KEYFRAME_DIR = "data/keyframes/"

def generate_video_filename(url: str) -> str:
    """
    Generate a unique filename based on the video URL.
    """
    video_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    return f"{video_hash}.mp4"

def generate_keyframe_directory(url: str) -> str:
    """
    Generate a directory for the keyframes based on the video URL hash.
    """
    video_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    return os.path.join(KEYFRAME_DIR, video_hash)  # Directory named with the video hash

def format_timestamp(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

def check_existing_keyframes(url: str) -> bool:
    """
    Checks if the keyframe directory already exists for the given video URL.
    """
    keyframe_directory = generate_keyframe_directory(url)
    
    # Check if the directory exists
    if os.path.exists(keyframe_directory):
        return True
    
    return False

def extract_keyframes_from_url(url: str, interval_sec=2):
    """
    Extracts keyframes from the hashed video file derived from the given URL.
    """
    video_filename = generate_video_filename(url)
    video_path = os.path.join(VIDEO_DIR, video_filename)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at {video_path}. Make sure it was downloaded first.")

    # Check if keyframes already exist for this video
    if check_existing_keyframes(url):
        print(f"Keyframes already extracted for {url}. Skipping extraction.")
        return

    os.makedirs(KEYFRAME_DIR, exist_ok=True)
    keyframe_directory = generate_keyframe_directory(url)
    os.makedirs(keyframe_directory, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    frame_interval = int(fps * interval_sec)

    keyframes = []

    print(f"Extracting keyframes every {interval_sec}s from {video_filename}...")

    frame_num = 0
    while frame_num < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_num / fps
        timestamp_hms = format_timestamp(timestamp_sec)
        filename = f"frame_{int(timestamp_sec):04d}.jpg"
        filepath = os.path.join(keyframe_directory, filename)

        cv2.imwrite(filepath, frame)
        keyframes.append({
            "timestamp_sec": round(timestamp_sec, 2),
            "timestamp_hms": timestamp_hms,
            "filename": filename
        })

        frame_num += frame_interval

    cap.release()

    # Save the keyframes metadata to a file within the keyframe directory
    metadata_filename = os.path.join(keyframe_directory, "keyframes_metadata.json")
    with open(metadata_filename, "w", encoding="utf-8") as f:
        json.dump(keyframes, f, indent=2)

    print(f"Saved {len(keyframes)} keyframes to {keyframe_directory} and metadata to {metadata_filename}")

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=dARr3lGKwk8"
    extract_keyframes_from_url(youtube_url)
