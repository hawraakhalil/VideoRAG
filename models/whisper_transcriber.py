import whisper
import os
import json
import yt_dlp
from datetime import timedelta
import subprocess
import hashlib
import time

# Define constants for the output directories and file paths
OUTPUT_DIR = "data/transcripts/" # directory where transcript segments will be saved
VIDEO_DIR = "data/raw_video/" # default path where the downloaded video will be saved
SEGMENT_FILE = os.path.join(OUTPUT_DIR, "segments.json") # path where the transcript segments (in JSON format) will be saved

def generate_video_filename(url: str) -> str:
    """
    Generate a unique filename based on the video URL.

    Args:
        url (str): The URL of the YouTube video.

    Returns:
        str: A file name generated from the URL's hash.
    """
    video_hash = hashlib.md5(url.encode('utf-8')).hexdigest()  # Generate a hash of the URL
    return f"{video_hash}.mp4"  # Return the filename based on the URL hash

def generate_segments_filename(url: str) -> str:
    """
    Generate a unique filename for the segments file based on the video URL hash.

    Args:
        url (str): The URL of the YouTube video.

    Returns:
        str: A file name for the segments based on the URL's hash.
    """
    video_hash = hashlib.md5(url.encode('utf-8')).hexdigest()  # Generate a hash of the URL
    return os.path.join(OUTPUT_DIR, f"{video_hash}_segments.json")  # Segments file with the video hash


def download_youtube_video(url: str, output_path: str = None):
    """
    Downloads a YouTube video using yt-dlp and saves it to the specified output path.
    If the video already exists, it skips the download.

    Args:
        url (str): The URL of the YouTube video to download.
        output_path (str): The path where the video will be saved. If None, the filename is generated from the URL.

    Returns: 
        str: The path where the video has been saved.
    """
    if output_path is None:
        output_path = os.path.join(VIDEO_DIR, generate_video_filename(url))  # Set default path in 'data/raw_video/'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # check that the output path exists, create it if necessary
    
    # Check if the video already exists
    if os.path.exists(output_path):
        print(f"Video already exists at {output_path}. Skipping download.")
        return output_path  # Skip downloading if the file already exists
    
    # Set options for yt-dlp
    ydl_opts = {
        'format': 'best',  # download the best quality
        'outtmpl': output_path  # save the video with the generated filename
    }
    
    # Retry mechanism in case of timeout errors
    retries = 3
    for attempt in range(retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])  # Download the video
            print(f"Video downloaded and saved to {output_path}")
            return output_path
        except Exception as e:
            if attempt < retries - 1:
                print(f"Error occurred: {e}. Retrying... ({attempt + 1}/{retries})")
                time.sleep(5)  # Wait before retrying
            else:
                print(f"Failed to download video after {retries} attempts.")
                raise e


def format_timestamp(seconds: float) -> str:
    """
    Converts seconds into a formatted timestamp string (HH:MM:SS).

    Args:
        seconds (float): The time in seconds to convert.

    Returns:
        str: The formatted timestamp (HH:MM:SS).
    """
    return str(timedelta(seconds=int(seconds)))

def transcribe_video(video_path: str, model_size: str = "base"):
    """
    Transcribes the audio from the given video using the Whisper model.

    Args:
        video_path (str): The path to the video to transcribe.
        model_size (str): The size of the Whisper model to use. Default is 'base'. Options: 'small', 'medium', 'large'.

    Returns:
        list: A list of segments where each segment is a dictionary with 'start', 'end', and 'text'.
    """
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    print("Transcribing...")
    result = model.transcribe(video_path, language="en")
    return result["segments"]  # each segment has 'start', 'end', 'text'

def segment_transcript(segments, chunk_size=10):
    """
    Segments the transcript into chunks based on the given chunk size (in seconds).

    Args:
        segments (list): The list of segments from the Whisper transcription, each containing 'start', 'end', and 'text'.
        chunk_size (int): The maximum duration (in seconds) of each chunk. Default is 10 seconds.

    Returns:
        list: A list of segmented transcripts, where each segment is a dictionary with 'start', 'end', 'text', 'start_hms', 'end_hms'.
    """
    segmented = []
    current_chunk = {
        "start": None,
        "end": None,
        "text": ""
    }

    # loop through each segment and build chunks
    for seg in segments:
        # start a new chunk if it's the first segment
        if current_chunk["start"] is None:
            current_chunk["start"] = seg["start"]

        # update the end time and accumulate the text for the current chunk
        current_chunk["end"] = seg["end"]
        current_chunk["text"] += " " + seg["text"]

        # calculate the duration of the current chunk
        duration = current_chunk["end"] - current_chunk["start"]

        # if the chunk exceeds the desired size, finalize this chunk and start a new one
        if duration >= chunk_size:
            segmented.append({
                "start": current_chunk["start"],
                "end": current_chunk["end"],
                "text": current_chunk["text"].strip(),
                "start_hms": format_timestamp(current_chunk["start"]),
                "end_hms": format_timestamp(current_chunk["end"])
            })
            current_chunk = {
                "start": None,
                "end": None,
                "text": ""
            }
    return segmented

def save_segments(segments, video_url: str):
    """
    Saves the segmented transcript to a JSON file named after the video hash.

    Args:
        segments (list): The list of segmented transcripts to save.
        video_url (str): The URL of the video to generate the filename for the segments.
    """
    segment_filename = generate_segments_filename(video_url)
    os.makedirs(os.path.dirname(segment_filename), exist_ok=True)
    with open(segment_filename, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"Saved transcript segments to {segment_filename}")


def load_existing_segments(video_url: str):
    """
    Checks if the segment file for the given video URL already exists.
    If it exists, loads the segments from the file.

    Args:
        video_url (str): The URL of the video to check for an existing segments file.

    Returns:
        list or None: A list of segments if the file exists, otherwise None.
    """
    segment_filename = generate_segments_filename(video_url)
    if os.path.exists(segment_filename):
        print(f"Segments already exist at {segment_filename}. Loading existing segments.")
        with open(segment_filename, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return None

def process_video(youtube_url: str):
    """
    Processes the video by checking if segments exist, and running transcription/segmentation if needed.

    Args:
        youtube_url (str): The URL of the YouTube video to process.
    """
    # Check if the segment file exists and load or process accordingly
    existing_segments = load_existing_segments(youtube_url)
    
    if existing_segments is not None:
        segmented = existing_segments
    else:
        video_path = download_youtube_video(youtube_url)
        whisper_segments = transcribe_video(video_path)
        segmented = segment_transcript(whisper_segments, chunk_size=10)
        save_segments(segmented, youtube_url)
    
    return segmented

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=dARr3lGKwk8"
    process_video(youtube_url)