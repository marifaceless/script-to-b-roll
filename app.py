import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
import requests
import time
import re
import os
import zipfile
import io
import base64
from typing import List, Dict, Any
import math
import tempfile
import subprocess
import shutil
import uuid
from PIL import Image
import numpy as np
import random

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize session state for persisting data between reruns
if 'selected_videos' not in st.session_state:
    st.session_state.selected_videos = {}
if 'segments' not in st.session_state:
    st.session_state.segments = []
if 'searched' not in st.session_state:
    st.session_state.searched = False
if 'segment_videos' not in st.session_state:
    st.session_state.segment_videos = {}
if 'all_videos' not in st.session_state:
    st.session_state.all_videos = []
if 'zip_data' not in st.session_state:
    st.session_state.zip_data = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'api_retry_count' not in st.session_state:
    st.session_state.api_retry_count = 0

# API key - use environment variable or secret
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "hbweXH4FwEDRICQK1bl0SyYygAOlHDeKZJc9ZPL6HOFym1NxXQNeNE9S")

# Political/city/country themed topics to bias search results
POLITICAL_THEMES = ["city", "urban", "government", "political", "country", "nation", 
                    "capital", "downtown", "skyline", "politics", "international", 
                    "global", "metropolis", "cityscape", "buildings", "architecture"]

# Rate limiting parameters
MAX_RETRIES = 5
BASE_DELAY = 2  # Initial delay in seconds
MAX_DELAY = 60  # Maximum delay in seconds

def parse_script_into_segments(script_text: str, segment_duration: int = 5) -> List[str]:
    """
    Parse script into segments of approximately 3-5 seconds each.
    Assumes average reading speed of ~150 words per minute or ~2.5 words per second.
    """
    # Clean script text
    script_text = re.sub(r'\s+', ' ', script_text).strip()
    
    # First split by sentences
    sentences = sent_tokenize(script_text)
    
    segments = []
    current_segment = ""
    word_count = 0
    words_per_segment = segment_duration * 2.5  # Words per segment based on reading speed
    
    for sentence in sentences:
        sentence_words = sentence.split()
        
        if word_count + len(sentence_words) <= words_per_segment:
            # Add whole sentence if it fits in current segment
            if current_segment:
                current_segment += " "
            current_segment += sentence
            word_count += len(sentence_words)
        else:
            # Check if we should split the sentence or start a new segment
            if word_count > 0:  # If we already have content in the segment
                segments.append(current_segment.strip())
                current_segment = sentence
                word_count = len(sentence_words)
            else:
                # Split long sentence into smaller chunks
                words = sentence_words
                while words:
                    chunk = words[:int(words_per_segment)]
                    segments.append(" ".join(chunk))
                    words = words[int(words_per_segment):]
    
    # Add the last segment if not empty
    if current_segment:
        segments.append(current_segment.strip())
    
    return segments

def extract_keywords(segment: str) -> List[str]:
    """Extract key keywords from a segment."""
    # This is a simple implementation - could be improved with NLP libraries
    # Remove common stopwords and punctuation
    stopwords = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "is", "are"]
    
    words = re.findall(r'\b\w+\b', segment.lower())
    keywords = [word for word in words if word not in stopwords and len(word) > 3]
    
    # Return top 3 keywords or all if less than 3
    keywords = keywords[:3] if len(keywords) > 3 else keywords
    
    # Add a political/city theme bias if no strong keywords found
    if len(keywords) < 2:
        keywords.append(POLITICAL_THEMES[hash(segment) % len(POLITICAL_THEMES)])
    
    return keywords

def is_16_9_aspect_ratio(width: int, height: int, tolerance: float = 0.1) -> bool:
    """Check if the video has approximately 16:9 aspect ratio with some tolerance."""
    target_ratio = 16 / 9
    actual_ratio = width / height if height != 0 else 0
    return abs(actual_ratio - target_ratio) <= tolerance

def calculate_video_score(video: Dict, segment: str) -> float:
    """Calculate a score for a video based on various factors to determine best fit."""
    score = 0.0
    
    # Prefer videos with duration close to 5 seconds
    duration = video.get("duration", 0)
    if 3 <= duration <= 5:
        score += 5.0  # Perfect duration
    elif 5 < duration <= 10:
        score += 3.0  # Good duration that can be trimmed
    elif duration > 10:
        score += 1.0  # Can be trimmed but not ideal
        
    # Higher resolution is better
    width = video.get("width", 0)
    if width >= 1920:
        score += 2.0  # Full HD or better
    elif width >= 1280:
        score += 1.5  # HD
        
    # Prefer videos with certain keywords in title/description
    video_text = (video.get("user", {}).get("name", "") + " " + 
                 video.get("url", "")).lower()
    
    # Check if video contains political/city keywords
    for theme in POLITICAL_THEMES:
        if theme in video_text:
            score += 0.5
            
    # Check if segment keywords are in video text
    segment_keywords = extract_keywords(segment)
    for keyword in segment_keywords:
        if keyword in video_text:
            score += 1.0
            
    return score

def api_request_with_backoff(url: str, headers: Dict[str, str]) -> Dict:
    """Make API request with exponential backoff for rate limiting."""
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            # Add small jitter to avoid synchronized requests when processing multiple segments
            time.sleep(random.uniform(1.5, 3.0) * (retry_count + 1))
            
            response = requests.get(url, headers=headers)
            
            # If successful, return the data
            if response.status_code == 200:
                return response.json()
                
            # If rate limited, wait and retry
            if response.status_code == 429:
                # Calculate backoff time with exponential increase and jitter
                delay = min(BASE_DELAY * (2 ** retry_count) + random.uniform(0, 1), MAX_DELAY)
                st.warning(f"Rate limit reached. Waiting {delay:.1f} seconds before retrying...")
                time.sleep(delay)
                retry_count += 1
                st.session_state.api_retry_count += 1
                continue
                
            # For other errors, raise exception
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            # If we've exceeded retries or it's not a rate limit issue, raise error
            if retry_count >= MAX_RETRIES - 1 or "429" not in str(e):
                raise e
            
            # For rate limit errors, retry with backoff
            delay = min(BASE_DELAY * (2 ** retry_count) + random.uniform(0, 1), MAX_DELAY)
            st.warning(f"Rate limit reached. Waiting {delay:.1f} seconds before retrying...")
            time.sleep(delay)
            retry_count += 1
            st.session_state.api_retry_count += 1
    
    # If we've exhausted retries
    raise Exception(f"Failed after {MAX_RETRIES} retries due to rate limiting")

def search_pexels_videos(query: str, per_page: int = 8, segment: str = "") -> List[Dict[Any, Any]]:
    """Search Pexels API for 16:9 videos matching the query with political/city bias."""
    # Add political/city bias to search query
    biased_query = query
    if not any(theme in query.lower() for theme in POLITICAL_THEMES):
        # Add a random political theme if none present
        bias_term = POLITICAL_THEMES[hash(query) % len(POLITICAL_THEMES)]
        biased_query = f"{query} {bias_term}"
    
    url = f"https://api.pexels.com/videos/search?query={biased_query}&per_page={per_page}"
    headers = {"Authorization": PEXELS_API_KEY}
    
    try:
        # Use backoff strategy for API request
        data = api_request_with_backoff(url, headers)
        videos = data.get("videos", [])
        
        # Filter for 16:9 videos only
        filtered_videos = []
        for video in videos:
            width = video.get("width", 0)
            height = video.get("height", 0)
            
            if is_16_9_aspect_ratio(width, height):
                # Filter for videos shorter than 15 seconds (easier to trim to 3-5s)
                if video.get("duration", 0) <= 15:
                    filtered_videos.append(video)
        
        # If we don't have enough 16:9 videos and have a next page, get more
        # But only if we haven't hit too many rate limits
        if len(filtered_videos) < 3 and "next_page" in data and st.session_state.api_retry_count < 5:
            try:
                more_videos = search_pexels_videos_by_url(data["next_page"], headers)
                for video in more_videos:
                    width = video.get("width", 0)
                    height = video.get("height", 0)
                    if is_16_9_aspect_ratio(width, height) and video.get("duration", 0) <= 15:
                        filtered_videos.append(video)
                        if len(filtered_videos) >= 4:  # Get enough videos to choose from
                            break
            except Exception as e:
                # If getting more videos fails, just continue with what we have
                st.warning(f"Couldn't get additional videos: {str(e)}")
        
        # Score and sort videos for best fit if segment is provided
        if segment:
            for video in filtered_videos:
                video["score"] = calculate_video_score(video, segment)
            filtered_videos.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return filtered_videos[:4]  # Return at most 4 videos
    except Exception as e:
        st.error(f"Error fetching videos: {str(e)}")
        return []

def search_pexels_videos_by_url(url: str, headers: Dict[str, str]) -> List[Dict[Any, Any]]:
    """Search Pexels API using a specific URL (for pagination)."""
    try:
        # Use backoff strategy for API request
        data = api_request_with_backoff(url, headers)
        return data.get("videos", [])
    except Exception as e:
        st.warning(f"Error in pagination request: {str(e)}")
        return []

def download_video(url: str) -> bytes:
    """Download video from URL."""
    # Add retry logic for downloads too
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.get(url, timeout=30)  # Add timeout
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed to download video after {max_retries} attempts: {str(e)}")
            
            # Wait before retrying
            delay = 2 ** retry_count  # Exponential backoff
            time.sleep(delay)
    
    # Should not reach here, but just in case
    raise Exception("Failed to download video for unknown reason")

def trim_video(video_data: bytes, start_time: float = 0, duration: float = 5) -> bytes:
    """Trim the video to the specified duration using FFmpeg."""
    try:
        # Create temporary files
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, "input.mp4")
        output_path = os.path.join(temp_dir, "output.mp4")
        
        # Write input video to temp file
        with open(input_path, "wb") as f:
            f.write(video_data)
        
        # Trim the video using FFmpeg
        cmd = [
            "ffmpeg", 
            "-i", input_path, 
            "-ss", str(start_time), 
            "-t", str(duration), 
            "-c:v", "libx264", 
            "-c:a", "aac", 
            "-strict", "experimental", 
            "-b:a", "128k", 
            output_path
        ]
        
        # Run command with suppressed output
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            # If FFmpeg fails, return the original video
            st.warning("Video trimming failed with FFmpeg. Using original video.")
            return video_data
        
        # Read the trimmed video
        with open(output_path, "rb") as f:
            trimmed_data = f.read()
        
        # Clean up temp files
        shutil.rmtree(temp_dir)
        
        return trimmed_data
    except Exception as e:
        # If anything goes wrong, return the original video
        st.warning(f"Video trimming failed: {str(e)}. Using original video.")
        return video_data

def create_zip_file(video_data: List[Dict[str, bytes]]) -> bytes:
    """Create a zip file containing all downloaded videos."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for video in video_data:
            zip_file.writestr(video["filename"], video["data"])
    
    return zip_buffer.getvalue()

def get_download_link(zip_data: bytes, filename: str) -> str:
    """Generate a download link for the zip file."""
    b64_data = base64.b64encode(zip_data).decode()
    href = f'<a href="data:application/zip;base64,{b64_data}" download="{filename}" class="download-button">Download B-Roll ZIP File</a>'
    return href

def auto_select_best_video(segment_key: str, videos: List[Dict[Any, Any]]) -> Dict[Any, Any]:
    """Automatically select the best video for a segment."""
    if not videos:
        return None
    
    # Sort videos by score (highest first)
    sorted_videos = sorted(videos, key=lambda x: x.get("score", 0), reverse=True)
    
    # Return the best video
    return sorted_videos[0] if sorted_videos else None

def process_script(script_text: str, segment_duration: int) -> None:
    """Process the entire script and automatically prepare videos for download."""
    # Reset retry counter at the start of processing
    st.session_state.api_retry_count = 0
    
    # Parse script into segments
    segments = parse_script_into_segments(script_text, segment_duration)
    st.session_state.segments = segments
    
    # Reset previously selected videos
    st.session_state.selected_videos = {}
    st.session_state.segment_videos = {}
    st.session_state.all_videos = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create container for results
    results_container = st.container()
    
    with results_container:
        st.subheader("B-Roll Processing Results")
        
        for i, segment in enumerate(segments):
            # Update progress
            progress = (i + 1) / (len(segments) * 2)  # First half for searching
            progress_bar.progress(progress)
            status_text.text(f"Finding videos for segment {i+1} of {len(segments)}")
            
            # Extract keywords from segment
            keywords = extract_keywords(segment)
            query = " ".join(keywords)
            
            # Search for videos with segment for scoring
            videos = search_pexels_videos(query, segment=segment)
            
            # Store segment key
            segment_key = f"segment_{i+1}"
            
            # Store videos for this segment
            st.session_state.segment_videos[segment_key] = videos
            
            # Automatically select best video
            best_video = auto_select_best_video(segment_key, videos)
            
            if best_video:
                # Find the best video file (HD if available)
                video_files = best_video.get("video_files", [])
                video_url = ""
                
                # Try to find HD file first
                for file in video_files:
                    if file.get("quality") == "hd":
                        video_url = file.get("link", "")
                        break
                
                # If no HD file, get the first one
                if not video_url and video_files:
                    video_url = video_files[0].get("link", "")
                
                # Create video info dictionary
                video_info = {
                    "segment": i+1,
                    "segment_text": segment,
                    "video_id": 1,  # Always 1 since we're only selecting the best
                    "url": video_url,
                    "filename": f"segment_{i+1}.mp4",
                    "duration": best_video.get("duration", 0),
                    "score": best_video.get("score", 0),
                    "image_url": best_video.get("image", ""),
                    "pexels_url": best_video.get("url", ""),
                    "width": best_video.get("width", 0),
                    "height": best_video.get("height", 0)
                }
                
                # Add to selected videos
                st.session_state.selected_videos[segment_key] = [video_info]
                st.session_state.all_videos.append(video_info)
            
            # Sleep between segments to avoid hitting rate limits
            # Add longer sleep if we've had API retry issues
            sleep_time = 2.0 + (st.session_state.api_retry_count * 0.5)
            time.sleep(sleep_time)
        
        # Second phase: downloading and trimming
        video_data = []
        
        for i, video_info in enumerate(st.session_state.all_videos):
            # Update progress for download phase
            progress = 0.5 + ((i + 1) / (len(st.session_state.all_videos) * 2))  # Second half for downloading
            progress_bar.progress(progress)
            status_text.text(f"Processing video {i+1} of {len(st.session_state.all_videos)}")
            
            try:
                # Download video
                data = download_video(video_info["url"])
                
                # Trim video to match segment duration
                duration = video_info.get("duration", 0)
                if duration > segment_duration:
                    # Start after 10% of the video to skip intros
                    start_time = min(duration * 0.1, duration - segment_duration)
                else:
                    start_time = 0
                
                # Trim to segment_duration or video duration, whichever is shorter
                trim_duration = min(segment_duration, duration)
                
                # Trim the video
                data = trim_video(data, start_time, trim_duration)
                
                # Store video data
                video_data.append({
                    "filename": video_info["filename"],
                    "data": data
                })
                
                # Sleep between downloads to avoid overloading
                time.sleep(1.0)
                
            except Exception as e:
                st.error(f"Error processing video for segment {i+1}: {str(e)}")
        
        # Create zip file
        status_text.text("Creating ZIP file...")
        zip_data = create_zip_file(video_data)
        
        # Store zip data in session state
        st.session_state.zip_data = zip_data
        
        # Complete progress
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Mark as processed
        st.session_state.searched = True
        st.session_state.processing_complete = True

# Main App
st.set_page_config(page_title="Script to B-Roll Finder", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
.download-button {
    display: inline-block;
    padding: 12px 24px;
    background-color: #4CAF50;
    color: white;
    text-decoration: none;
    font-weight: bold;
    border-radius: 4px;
    text-align: center;
    transition: background-color 0.3s;
    margin: 20px 0;
}
.download-button:hover {
    background-color: #45a049;
}
.stButton>button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.title("Automated Script to B-Roll Finder")
st.markdown("Paste your script below to automatically find and download B-roll footage for each segment")

# User inputs
script_text = st.text_area("Paste your script here:", height=200)
segment_duration = st.slider("Segment duration (seconds):", 3, 5, 4, help="Target duration for each script segment and corresponding B-roll clip")

# Reset button to start over
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Reset", help="Clear all results and start over"):
        st.session_state.selected_videos = {}
        st.session_state.segments = []
        st.session_state.searched = False
        st.session_state.segment_videos = {}
        st.session_state.all_videos = []
        st.session_state.zip_data = None
        st.session_state.processing_complete = False
        st.session_state.api_retry_count = 0
        st.experimental_rerun()

with col2:
    if st.button("Process Script & Find B-Roll", type="primary", help="Automatically find, trim, and package B-roll footage for your script") and script_text:
        process_script(script_text, segment_duration)

# Display download section if processing is complete
if st.session_state.processing_complete and st.session_state.zip_data:
    st.success(f"🎉 Successfully found and processed B-roll for {len(st.session_state.all_videos)} script segments!")
    
    # Display download button
    st.markdown(get_download_link(st.session_state.zip_data, "broll_footage.zip"), unsafe_allow_html=True)
    
    # Display summary of selected videos
    with st.expander("View Selected B-Roll Details", expanded=False):
        st.subheader("Selected B-Roll Videos")
        
        for i, video_info in enumerate(st.session_state.all_videos):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image(video_info.get("image_url", ""), use_column_width=True)
            
            with col2:
                st.markdown(f"### Segment {i+1}")
                st.text(video_info.get("segment_text", ""))
                st.write(f"**Duration:** {video_info.get('duration', 0)}s (trimmed to {segment_duration}s)")
                st.write(f"**Resolution:** {video_info.get('width', 0)}x{video_info.get('height', 0)}")
                st.write(f"**Match Score:** {video_info.get('score', 0):.1f}")
                st.markdown(f"[View on Pexels]({video_info.get('pexels_url', '')})")
            
            st.divider()

# Show appropriate message if script is being processed
elif st.session_state.searched and not st.session_state.processing_complete:
    st.info("Still processing your script. Please wait...")

# Show instructions for first-time users
elif not st.session_state.searched:
    st.markdown("""
    ### How to use this app:
    1. Paste your script in the text area above
    2. Adjust the segment duration if needed (3-5 seconds)
    3. Click "Process Script & Find B-Roll"
    4. Wait for processing to complete (it may take a few minutes)
    5. Download the ZIP file containing all B-roll videos

    #### What happens behind the scenes:
    - Your script is broken into segments based on estimated speaking time
    - Each segment is analyzed to extract key topics
    - The app finds the most relevant 16:9 videos for each segment
    - Videos are automatically trimmed to match your segment duration
    - All videos are packaged into a single ZIP file for easy download
    
    #### Note:
    This app uses the Pexels API to find free stock footage. Make sure to credit Pexels and the video creators when using the footage.
    """)

# Add note about API rate limiting
if st.session_state.api_retry_count > 0:
    st.warning(f"Note: API rate limiting was encountered {st.session_state.api_retry_count} times during processing. If you experience issues, please try again later or with a smaller script.")

# Footer with attribution
st.markdown("---")
st.caption("Powered by Pexels API | Stock footage provided by Pexels.com") 
