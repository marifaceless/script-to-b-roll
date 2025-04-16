import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
import requests
import time
import re
from typing import List, Dict, Any
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# API key - use environment variable or secret
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "hbweXH4FwEDRICQK1bl0SyYygAOlHDeKZJc9ZPL6HOFym1NxXQNeNE9S")

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
    return keywords[:3] if len(keywords) > 3 else keywords

def search_pexels_videos(query: str, per_page: int = 4) -> List[Dict[Any, Any]]:
    """Search Pexels API for videos matching the query."""
    url = f"https://api.pexels.com/videos/search?query={query}&per_page={per_page}"
    headers = {"Authorization": PEXELS_API_KEY}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("videos", [])
    except Exception as e:
        st.error(f"Error fetching videos: {str(e)}")
        return []

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
                cols = st.columns(min(4, len(videos)))
                for j, video in enumerate(videos[:4]):
                    with cols[j]:
                        # Get the smallest preview image
                        image_url = video.get("image", "")
                        
                        # Display thumbnail
                        st.image(image_url, use_column_width=True)
                        
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
                        st.markdown(f"[Download Video]({download_url})")
                        
                        # Add link to Pexels page for attribution
                        st.markdown(f"[View on Pexels]({video.get('url', '')})")
            else:
                st.warning("No videos found for this segment. Try different keywords.")
            
            # Add a divider between segments
            st.divider()
            
            # Sleep to avoid hitting API rate limits
            time.sleep(0.5)
    
    # Complete progress bar
    progress_bar.empty()
    status_text.text("Processing complete!")
    
    st.success("B-roll suggestions generated successfully!")

st.markdown("---")
st.markdown("""
### How to use this app:
1. Paste your script in the text area
2. Adjust segment duration if needed (3-5 seconds)
3. Click "Find B-Roll Footage"
4. Review the suggestions and download the ones you like

#### Note:
This app uses the Pexels API to find B-roll footage. Make sure to credit Pexels and the creators when using the footage.
""") 