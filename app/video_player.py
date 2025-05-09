import streamlit as st
import os
import hashlib
from datetime import timedelta
from models.keyframe_extractor import generate_keyframe_directory

def get_video_path(url):
    """
    Generate the path to the video file based on the URL.
    
    Args:
        url (str): The YouTube URL of the video
        
    Returns:
        str: The path to the video file
    """
    video_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    # Create path relative to project root, not app directory
    video_path = os.path.join("data", "raw_video", f"{video_hash}.mp4")
    return video_path

def format_time(seconds):
    """Format seconds as HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def display_video(url, start_time=0):
    """
    Display a video in the Streamlit UI with an optional starting timestamp.
    
    Args:
        url (str): The YouTube URL of the video
        start_time (float, optional): The time in seconds where the video should start
        
    Returns:
        bool: True if the video was displayed successfully, False otherwise
    """
    video_path = get_video_path(url)
    
    if not os.path.exists(video_path):
        st.error(f"Video file not found at {video_path}")
        return False
    
    # Get absolute path to video
    abs_video_path = os.path.abspath(video_path)
    
    # Display timestamp information
    st.write(f"Video timestamp: {format_time(start_time)}")
    
    # Use Streamlit's native video display with start_time
    st.video(abs_video_path, start_time=int(start_time))
    
    # Let user know the video will play for 20 seconds
    st.info(f"Video will play from {format_time(start_time)} to {format_time(start_time + 20)}")
    
    return True

def display_video_segment(url, start_time, end_time=None, show_keyframe=False):
    """
    Display a specific segment of a video.
    
    Args:
        url (str): The YouTube URL of the video
        start_time (float): The start time of the segment in seconds
        end_time (float, optional): The end time of the segment in seconds
        show_keyframe (bool, optional): Whether to display a keyframe from the segment (default: False)
        
    Returns:
        bool: True if the segment was displayed successfully, False otherwise
    """
    # If end_time isn't provided, play for 20 seconds
    if end_time is None:
        end_time = start_time + 20
    
    # Calculate duration
    duration = end_time - start_time
    
    video_path = get_video_path(url)
    
    if not os.path.exists(video_path):
        st.error(f"Video file not found at {video_path}")
        return False
    
    # Get absolute path to video
    abs_video_path = os.path.abspath(video_path)
    
    # Display segment information
    st.write(f"Video segment: {format_time(start_time)} to {format_time(end_time)} (Duration: {format_time(duration)})")
    
    # Use Streamlit's native video display
    st.video(abs_video_path, start_time=int(start_time))
    
    return True

def create_timestamp_link(seconds, label=None):
    """
    Create a clickable timestamp link for navigating in the video.
    
    Args:
        seconds (float): The timestamp in seconds
        label (str, optional): The label for the link. If None, formats the seconds as HH:MM:SS
        
    Returns:
        str: HTML for a clickable timestamp
    """
    if label is None:
        # Format seconds as HH:MM:SS
        label = format_time(seconds)
    
    return f"[{label}]"  # Return a simple formatted string as Streamlit doesn't support HTML links here
