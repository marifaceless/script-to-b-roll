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

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# API key - use environment variable or secret
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "hbweXH4FwEDRICQK1bl0SyYygAOlHDeKZJc9ZPL6HOFym1NxXQNeNE9S")

# Political/city/country themed topics to bias search results
POLITICAL_THEMES = ["city", "urban", "government", "political", "country", "nation", 
                    "capital", "downtown", "skyline", "politics", "international", 
                    "global", "metropolis", "cityscape", "buildings", "architecture"]

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

def search_pexels_videos(query: str, per_page: int = 10) -> List[Dict[Any, Any]]:
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
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        videos = data.get("videos", [])
        
        # Filter for 16:9 videos only
        filtered_videos = []
        for video in videos:
            width = video.get("width", 0)
            height = video.get("height", 0)
            
            if is_16_9_aspect_ratio(width, height):
                filtered_videos.append(video)
        
        # If we don't have enough 16:9 videos, get more
        if len(filtered_videos) < 4 and "next_page" in data:
            more_videos = search_pexels_videos_by_url(data["next_page"], headers)
            for video in more_videos:
                width = video.get("width", 0)
                height = video.get("height", 0)
                if is_16_9_aspect_ratio(width, height):
                    filtered_videos.append(video)
                    if len(filtered_videos) >= 4:
                        break
        
        return filtered_videos[:4]  # Return at most 4 videos
    except Exception as e:
        st.error(f"Error fetching videos: {str(e)}")
        return []

def search_pexels_videos_by_url(url: str, headers: Dict[str, str]) -> List[Dict[Any, Any]]:
    """Search Pexels API using a specific URL (for pagination)."""
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("videos", [])
    except Exception:
        return []

def download_video(url: str) -> bytes:
    """Download video from URL."""
    response = requests.get(url)
    return response.content

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
    href = f'<a href="data:application/zip;base64,{b64_data}" download="{filename}">Download Zip File</a>'
    return href

# Main App
st.set_page_config(page_title="Script to B-Roll Finder", layout="wide")

st.title("Script to B-Roll Finder")
st.markdown("Upload or paste your script to find relevant B-roll footage")

# User inputs
script_text = st.text_area("Paste your script here:", height=200)
segment_duration = st.slider("Segment duration (seconds):", 3, 5, 4)

if st.button("Find B-Roll Footage") and script_text:
    # Parse script into segments
    segments = parse_script_into_segments(script_text, segment_duration)
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create container for results
    results_container = st.container()
    
    # Store all video data for download
    all_videos = []
    segment_videos = {}
    
    with results_container:
        st.subheader("B-Roll Suggestions")
        
        for i, segment in enumerate(segments):
            # Update progress
            progress = (i + 1) / len(segments)
            progress_bar.progress(progress)
            status_text.text(f"Processing segment {i+1} of {len(segments)}")
            
            st.markdown(f"### Segment {i+1}")
            st.text(segment)
            
            # Extract keywords from segment
            keywords = extract_keywords(segment)
            query = " ".join(keywords)
            st.caption(f"Search query: {query}")
            
            # Search for videos
            videos = search_pexels_videos(query)
            if videos:
                segment_videos[f"segment_{i+1}"] = []
                
                cols = st.columns(min(4, len(videos)))
                for j, video in enumerate(videos[:4]):
                    with cols[j]:
                        # Get the preview image
                        image_url = video.get("image", "")
                        
                        # Display thumbnail
                        st.image(image_url, use_column_width=True)
                        
                        # Get video dimensions for aspect ratio display
                        width = video.get("width", 0)
                        height = video.get("height", 0)
                        aspect_ratio = f"{width}:{height}"
                        
                        # Get best video file (prefer HD, but fallback to others)
                        video_files = video.get("video_files", [])
                        download_url = ""
                        
                        # Try to find HD file first
                        for file in video_files:
                            if file.get("quality") == "hd":
                                download_url = file.get("link", "")
                                break
                        
                        # If no HD file, get the first one
                        if not download_url and video_files:
                            download_url = video_files[0].get("link", "")
                        
                        # Video details
                        st.write(f"**Duration:** {video.get('duration', 0)}s")
                        st.write(f"**Aspect Ratio:** {aspect_ratio}")
                        
                        # Create selection checkbox
                        selected = st.checkbox(f"Select for segment {i+1}, video {j+1}")
                        if selected:
                            # Add to selected videos for download
                            video_info = {
                                "segment": i+1,
                                "video_id": j+1,
                                "url": download_url,
                                "filename": f"segment_{i+1}_video_{j+1}.mp4"
                            }
                            segment_videos[f"segment_{i+1}"].append(video_info)
                            all_videos.append(video_info)
                        
                        # Add link to Pexels page for attribution
                        st.markdown(f"[View on Pexels]({video.get('url', '')})")
            else:
                st.warning("No 16:9 videos found for this segment. Try different keywords.")
            
            # Add a divider between segments
            st.divider()
            
            # Sleep to avoid hitting API rate limits
            time.sleep(0.5)
    
    # Complete progress bar
    progress_bar.empty()
    status_text.text("Processing complete!")
    
    # If videos were selected, show download option
    if all_videos:
        st.success(f"Found {len(all_videos)} B-roll clips for your script!")
        
        if st.button("Prepare Download ZIP"):
            # Download progress
            download_progress = st.progress(0)
            download_status = st.empty()
            
            # Download all selected videos
            download_status.text("Downloading videos...")
            video_data = []
            
            for i, video_info in enumerate(all_videos):
                download_progress.progress((i + 1) / len(all_videos))
                download_status.text(f"Downloading video {i+1} of {len(all_videos)}...")
                
                try:
                    data = download_video(video_info["url"])
                    video_data.append({
                        "filename": video_info["filename"],
                        "data": data
                    })
                except Exception as e:
                    st.error(f"Error downloading video {video_info['filename']}: {str(e)}")
            
            # Create zip file
            download_status.text("Creating ZIP file...")
            zip_data = create_zip_file(video_data)
            
            # Create download link
            download_status.empty()
            download_progress.empty()
            
            st.markdown(get_download_link(zip_data, "broll_footage.zip"), unsafe_allow_html=True)
            st.success("ZIP file created! Click the link above to download.")
    else:
        st.info("No videos selected for download. Select videos to enable the download option.")

st.markdown("---")
st.markdown("""
### How to use this app:
1. Paste your script in the text area
2. Adjust segment duration if needed (3-5 seconds)
3. Click "Find B-Roll Footage"
4. Select the videos you want for each segment by checking the boxes
5. Click "Prepare Download ZIP" to download all selected videos in a zip file

#### Note:
This app uses the Pexels API to find B-roll footage with focus on 16:9 aspect ratio and political/city themes. Make sure to credit Pexels and the creators when using the footage.
""") 
