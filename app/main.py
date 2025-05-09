import streamlit as st
import os
import sys
import json
import time
import hashlib
import math

# Add project root to path to allow importing from other directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the required modules from the project
from models.whisper_transcriber import process_video, generate_segments_filename
from models.keyframe_extractor import extract_keyframes_from_url, generate_keyframe_directory
from models.text_embedding import generate_text_embeddings
from retrieval.retrieval_factory import RetrievalFactory, RetrievalMethod

# Import the app components
from app.chat_interface import (
    initialize_chat_state, 
    display_chat_history, 
    chat_input_section, 
    add_message_to_chat,
    clear_chat_history,
    create_sidebar_section
)
from app.video_player import display_video_segment

# Configure the Streamlit page
st.set_page_config(
    page_title="VideoRAG - Video Question Answering",
    page_icon="🎬",
    layout="wide",
)

# Load CSS
with open(os.path.join(os.path.dirname(__file__), "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state variables
initialize_chat_state()
if 'youtube_url' not in st.session_state:
    st.session_state.youtube_url = ""
if 'segments' not in st.session_state:
    st.session_state.segments = None
if 'video_loaded' not in st.session_state:
    st.session_state.video_loaded = False
if 'keyframes' not in st.session_state:
    st.session_state.keyframes = None
if 'retrieval_method' not in st.session_state:
    st.session_state.retrieval_method = RetrievalMethod.FAISS_FLAT
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {
        'transcript': {'status': '', 'progress': 0},
        'keyframes': {'status': '', 'progress': 0},
        'embeddings': {'status': '', 'progress': 0}
    }

# Define the confidence threshold for answers
CONFIDENCE_THRESHOLD = 0.7  # Adjusted for the new confidence scaling (0.5-1.0 range)

def load_segments(url):
    """Load transcript segments for a given video URL."""
    try:
        # Update processing status
        st.session_state.processing_status['transcript']['status'] = 'processing'
        
        # Process the video to get segments (will load existing if available)
        segments = process_video(url)
        st.session_state.segments = segments
        
        # Complete processing status
        st.session_state.processing_status['transcript']['status'] = 'complete'
        st.session_state.processing_status['transcript']['progress'] = 100
        return True
    except Exception as e:
        st.session_state.processing_status['transcript']['status'] = 'error'
        st.error(f"Error loading segments: {str(e)}")
        return False

def load_keyframes(url):
    """Extract keyframes from the video if needed and load metadata."""
    try:
        # Update processing status
        st.session_state.processing_status['keyframes']['status'] = 'processing'
        
        # Extract keyframes if not already done
        extract_keyframes_from_url(url)
        
        # Load keyframes metadata
        keyframe_dir = generate_keyframe_directory(url)
        metadata_path = os.path.join(keyframe_dir, "keyframes_metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                st.session_state.keyframes = json.load(f)
            
            # Complete processing status
            st.session_state.processing_status['keyframes']['status'] = 'complete'
            st.session_state.processing_status['keyframes']['progress'] = 100
            return True
        else:
            st.session_state.processing_status['keyframes']['status'] = 'error'
            st.error("Keyframe metadata not found")
            return False
    except Exception as e:
        st.session_state.processing_status['keyframes']['status'] = 'error'
        st.error(f"Error loading keyframes: {str(e)}")
        return False

def generate_embeddings(url):
    """Generate text embeddings for the video transcript if they don't exist."""
    try:
        # Update processing status
        st.session_state.processing_status['embeddings']['status'] = 'processing'
        
        # First check if embeddings already exist based on the selected method
        try:
            if st.session_state.retrieval_method == RetrievalMethod.TFIDF:
                from retrieval.tfidf_search import load_embeddings
            else:
                from retrieval.faiss_search import load_embeddings
                
            load_embeddings(url, modality="text")
            
            # Complete processing status
            st.session_state.processing_status['embeddings']['status'] = 'complete'
            st.session_state.processing_status['embeddings']['progress'] = 100
            
            st.success("✅ Using existing text embeddings")
            return True
        except FileNotFoundError:
            # Embeddings don't exist, generate them
            st.info("🔄 Generating text embeddings for the video...")
            
            # Make sure the transcript file exists
            segment_path = generate_segments_filename(url)
            if not os.path.exists(segment_path):
                st.error(f"Transcript file not found at: {segment_path}")
                # Copy segments from session state if available
                if st.session_state.segments:
                    st.warning("⚠️ Attempting to save segments from memory...")
                    try:
                        os.makedirs(os.path.dirname(segment_path), exist_ok=True)
                        with open(segment_path, "w", encoding="utf-8") as f:
                            json.dump(st.session_state.segments, f, indent=2, ensure_ascii=False)
                        st.success(f"✅ Successfully saved segments to {segment_path}")
                    except Exception as e:
                        st.session_state.processing_status['embeddings']['status'] = 'error'
                        st.error(f"❌ Failed to save segments: {str(e)}")
                        return False
                else:
                    st.session_state.processing_status['embeddings']['status'] = 'error'
                    return False
            
            # Now generate the embeddings based on the selected method
            if st.session_state.retrieval_method == RetrievalMethod.TFIDF:
                from retrieval.tfidf_search import fit_tfidf_vectorizer
                with open(segment_path, 'r', encoding='utf-8') as f:
                    segments = json.load(f)
                fit_tfidf_vectorizer(segments)
            elif st.session_state.retrieval_method == RetrievalMethod.BM25:
                from retrieval.bm25_search import fit_bm25_index, BM25_MODEL_PATH, BM25_DATA_PATH
                st.info("🔄 Generating BM25 index...")
                
                # Delete existing BM25 model files if they exist
                if os.path.exists(BM25_MODEL_PATH):
                    try:
                        os.remove(BM25_MODEL_PATH)
                        print(f"Deleted existing BM25 model file: {BM25_MODEL_PATH}")
                    except Exception as e:
                        print(f"Failed to delete {BM25_MODEL_PATH}: {str(e)}")
                
                if os.path.exists(BM25_DATA_PATH):
                    try:
                        os.remove(BM25_DATA_PATH)
                        print(f"Deleted existing BM25 data file: {BM25_DATA_PATH}")
                    except Exception as e:
                        print(f"Failed to delete {BM25_DATA_PATH}: {str(e)}")
                
                # Generate the correct segment path using URL hash
                url_hash = hashlib.md5(url.encode()).hexdigest()
                segment_path = f"data/transcripts/{url_hash}_segments.json"
                print(f"app/main.py: Looking for segments at: {segment_path}")
                
                with open(segment_path, 'r', encoding='utf-8') as f:
                    segments = json.load(f)
                print(f"app/main.py: Creating BM25 index with {len(segments)} segments")
                bm25, corpus = fit_bm25_index(segments)
                print(f"app/main.py: BM25 index created with {len(corpus)} documents")
                st.info(f"✅ BM25 index created with {len(corpus)} documents")
            else:
                # Default to FAISS or other embedding methods
                generate_text_embeddings(url)
            
            # Complete processing status
            st.session_state.processing_status['embeddings']['status'] = 'complete'
            st.session_state.processing_status['embeddings']['progress'] = 100
            
            st.success("✅ Text embeddings generated successfully")
            return True
    except Exception as e:
        st.session_state.processing_status['embeddings']['status'] = 'error'
        st.error(f"❌ Error generating embeddings: {str(e)}")
        st.error("Details: " + str(e))
        return False

def get_video_embeddings(url, retrieval_method=RetrievalMethod.FAISS_FLAT):
    """Load and prepare embeddings for search using the selected method."""
    try:
        print(f"get_video_embeddings: Loading embeddings with {retrieval_method.value}")
        
        # BM25 and TFIDF are special cases that should be handled separately
        if retrieval_method == RetrievalMethod.BM25:
            try:
                print("get_video_embeddings: Loading BM25 index...")
                from retrieval.bm25_search import load_embeddings, build_bm25_index
                corpus = load_embeddings(url, modality="text")
                index = build_bm25_index(corpus)
                print(f"get_video_embeddings: Successfully loaded BM25 index")
                return index, None
            except Exception as e:
                print(f"get_video_embeddings: Error loading BM25 index - {str(e)}")
                # Try to recreate BM25 index
                try:
                    print("get_video_embeddings: Attempting to regenerate BM25 index...")
                    from retrieval.bm25_search import fit_bm25_index
                    import hashlib
                    
                    # Generate the correct segment path using URL hash
                    url_hash = hashlib.md5(url.encode()).hexdigest()
                    segment_path = f"data/transcripts/{url_hash}_segments.json"
                    print(f"get_video_embeddings: Looking for segments at: {segment_path}")
                    
                    with open(segment_path, 'r', encoding='utf-8') as f:
                        segments = json.load(f)
                    bm25, _ = fit_bm25_index(segments)
                    print("get_video_embeddings: Successfully regenerated BM25 index")
                    return bm25, None
                except Exception as e2:
                    print(f"get_video_embeddings: Failed to regenerate BM25 index - {str(e2)}")
                    raise e
        
        # Load text embeddings with the selected method
        factory = RetrievalFactory()
        text_index = factory.load_embeddings(url, modality="text", method=retrieval_method)
        
        # Try to load image embeddings if available (but skip for TF-IDF and BM25 which only support text)
        image_index = None
        if retrieval_method not in [RetrievalMethod.TFIDF, RetrievalMethod.BM25]:
            try:
                image_index = factory.load_embeddings(url, modality="image", method=retrieval_method)
            except FileNotFoundError:
                image_index = None
        
        return text_index, image_index
    except Exception as e:
        print(f"get_video_embeddings: Error - {str(e)}")
        raise e

def search_video_content(query, url, retrieval_method=RetrievalMethod.FAISS_FLAT):
    """Search for relevant content based on the query using the selected retrieval method."""
    if not st.session_state.segments:
        st.error("No transcript segments available. Please load a video first.")
        return []
    
    print(f"search_video_content: Using {retrieval_method.value} to search for: '{query}'")
    
    try:
        # Get embeddings and build indices
        text_index, image_index = get_video_embeddings(url, retrieval_method)
        
        # Fix: BM25 and TFIDF don't need a special NULL handling because they're handled differently
        if text_index is None and retrieval_method not in [RetrievalMethod.PGVECTOR_IVFFLAT, RetrievalMethod.TFIDF, RetrievalMethod.BM25]:
            print("search_video_content: No text index found!")
            return []
        
        print(f"search_video_content: Got embeddings, text_index type: {type(text_index)}")
        
        # Embed the query using the selected method
        factory = RetrievalFactory()
        query_emb = factory.embed_query(query, method=retrieval_method)
        print(f"search_video_content: Query embedded, type: {type(query_emb)}")
        
        # Search for relevant segments using the selected method
        # Always use "text" modality for TF-IDF and BM25 to prevent errors
        modality = "text"  # TF-IDF and BM25 only support text anyway
        
        print(f"search_video_content: Starting search with {retrieval_method.value}")
        indices, distances = factory.search(
            query_emb, url, modality=modality, method=retrieval_method, top_k=5
        )
        
        print(f"search_video_content: Got {len(indices)} results")
        if len(indices) == 0:
            print("search_video_content: No matches found!")
            return []
        
        # Return the top matches with their segments
        results = []
        segments_len = len(st.session_state.segments)
        print(f"search_video_content: Total segments: {segments_len}")
        
        for idx, dist in zip(indices, distances):
            print(f"search_video_content: Processing result - index: {idx}, distance: {dist}")
            if idx < segments_len:
                segment = st.session_state.segments[idx]
                
                # Calculate confidence based on the retrieval method
                if retrieval_method == RetrievalMethod.TFIDF:
                    # Special case for TF-IDF which tends to have high scores
                    # Apply more aggressive scaling to bring values down
                    confidence = min(dist, 1.0)  # Cap at 1.0
                    # Apply sigmoid-like transformation to spread out the values
                    confidence = 0.5 + (0.5 * (confidence - 0.5) / (0.3 + abs(confidence - 0.5)))
                    # Final range adjustment
                    confidence = max(0.5, min(0.95, confidence)) - 0.03
                elif retrieval_method == RetrievalMethod.BM25:
                    # For BM25, higher is better but the scale can vary widely
                    # First cap extremely high values
                    raw_score = min(dist, 10.0)  # BM25 can give very high scores, cap at 10.0
                    
                    # Apply a logarithmic scaling to compress the range
                    # This gives more differentiation in the lower scores
                    # log(1+x) maps 0→0, 1→0.69, 2→1.1, 5→1.8, 10→2.4
                    if raw_score > 0:
                        log_score = math.log1p(raw_score) / math.log1p(10.0)  # Normalize to 0-1 range
                    else:
                        log_score = 0
                    
                    # Map this to our confidence range with a bias toward lower values
                    # This prevents "false confidence" in weak matches
                    confidence = 0.5 + (0.4 * log_score)
                    
                    # Adjust to ensure good differentiation between scores
                    if raw_score < 1.0:  # Very low scores should have low confidence
                        confidence = max(0.5, confidence - 0.15)
                else:
                    # For vector distances (like FAISS), lower is better, so we invert
                    # The original distances might be very large, leading to near-zero confidence
                    # Apply a sigmoid-like scaling to keep values in a reasonable range
                    scaled_dist = min(dist, 2.0)  # Cap distance at 2.0
                    confidence = 1.0 - (scaled_dist / 2.0)  # Convert to 0.0-1.0 range
                    # Boost low confidences to prevent near-zero values
                    confidence = 0.5 + (0.5 * confidence)  # Range: 0.5-1.0
                
                print(f"search_video_content: Found segment with confidence {confidence:.2f}, original distance: {dist:.4f}, method: {retrieval_method.value}")
                results.append({
                    "segment": segment,
                    "distance": dist,
                    "confidence": confidence
                })
            else:
                print(f"search_video_content: WARNING - Index {idx} out of bounds (max: {segments_len-1})")
        
        # Sort by confidence (highest first)
        results.sort(key=lambda x: x["confidence"], reverse=True)
        print(f"search_video_content: Returning {len(results)} sorted results")
        
        return results
    except Exception as e:
        import traceback
        print(f"search_video_content: Error - {str(e)}")
        print(f"search_video_content: Traceback - {traceback.format_exc()}")
        st.error(f"Error searching video content: {str(e)}")
        return []

def handle_user_query(query):
    """Process the user query and update chat history."""
    # Skip empty queries
    if not query.strip():
        return
    
    # Add user message to chat
    add_message_to_chat("user", query)
    
    # Check if a video is loaded
    if not st.session_state.youtube_url or not st.session_state.video_loaded:
        add_message_to_chat("assistant", "⚠️ Please load a video first using the sidebar controls.")
        return
    
    # Starting the search
    with st.status("🔍 Searching for relevant information...", expanded=True) as status:
        try:
            # Get the current retrieval method from session state
            retrieval_method = st.session_state.retrieval_method
            
            # Search for relevant content using the selected method
            results = search_video_content(query, st.session_state.youtube_url, retrieval_method)
            
            if not results:
                add_message_to_chat("assistant", "⚠️ I couldn't find any relevant information in the video. Please try asking a different question.")
                status.update(label="❌ No relevant information found", state="error")
                return
            
            # Get the top result
            top_result = results[0]
            confidence = top_result["confidence"]
            
            status.update(label="✨ Generating response...", state="running")
            
            # Check if confidence meets our threshold
            if confidence < CONFIDENCE_THRESHOLD:
                # Not confident enough in the answer
                response = f"""I'm not confident that the answer to your question is present in this video.

The closest match (confidence: {confidence:.2f}) is at {top_result['segment']['start_hms']}:
"{top_result['segment']['text']}"

But this doesn't appear to directly answer your question. Could you try rephrasing?
"""
                add_message_to_chat("assistant", response)
                status.update(label="⚠️ Low confidence answer", state="error")
                return
            
            # We have a confident answer
            segment = top_result["segment"]
            
            # Build response with timestamp link for interaction
            timestamp_link = f"<span class='timestamp-link' title='Click to play this segment'>{segment['start_hms']} - {segment['end_hms']}</span>"
            
            response = f"""I found an answer:

{timestamp_link} "{segment['text']}"

<div class='info-box'>
<strong>Confidence:</strong> {confidence:.2f}
<br>
Click on the timestamp to play this segment of the video.
</div>
"""
            
            # Add assistant response to chat history with timestamp for video playback
            add_message_to_chat("assistant", response, timestamp=segment["start"])
            status.update(label="✅ Response generated successfully", state="complete")
            
        except Exception as e:
            import traceback
            print(f"handle_user_query: Exception - {str(e)}")
            print(f"handle_user_query: Traceback - {traceback.format_exc()}")
            status.update(label=f"❌ Error: {str(e)}", state="error")
            add_message_to_chat("assistant", f"⚠️ An error occurred while processing your question: {str(e)}")

def display_video_callback(message):
    """Callback to display video segment when assistant response is shown."""
    if "timestamp" in message:
        display_video_segment(st.session_state.youtube_url, message["timestamp"])

def setup_database_if_needed():
    """Set up the database connection if using PGVector."""
    if st.session_state.retrieval_method == RetrievalMethod.PGVECTOR_IVFFLAT:
        try:
            from retrieval.pgvector_search import setup_database_connection
            setup_database_connection()
        except Exception as e:
            st.error(f"❌ Error setting up database connection: {str(e)}")
            st.error("Make sure PostgreSQL is running and the pgvector extension is installed.")
            return False
    return True

def process_url_button_click():
    """Handle the Load Video button click."""
    # Clear processing status
    st.session_state.processing_status = {
        'transcript': {'status': '', 'progress': 0},
        'keyframes': {'status': '', 'progress': 0},
        'embeddings': {'status': '', 'progress': 0}
    }
    
    # Reset video loaded state
    st.session_state.video_loaded = False
    
    # Store the URL in session state
    st.session_state.youtube_url = st.session_state.url_input

def display_processing_status():
    """Display the current processing status in the sidebar."""
    status = st.session_state.processing_status
    
    # Create a status container
    status_container = st.sidebar.container()
    
    with status_container:
        st.markdown("### Processing Status")
        
        # Display transcript status
        st.markdown("#### Transcript")
        if status['transcript']['status'] == 'processing':
            st.progress(50, "⏳ Processing...")
        elif status['transcript']['status'] == 'complete':
            st.progress(100, "✅ Complete")
        elif status['transcript']['status'] == 'error':
            st.error("❌ Failed")
        else:
            st.progress(0, "🔄 Waiting...")
        
        # Display keyframes status
        st.markdown("#### Keyframes")
        if status['keyframes']['status'] == 'processing':
            st.progress(50, "⏳ Processing...")
        elif status['keyframes']['status'] == 'complete':
            st.progress(100, "✅ Complete")
        elif status['keyframes']['status'] == 'error':
            st.error("❌ Failed")
        else:
            st.progress(0, "🔄 Waiting...")
        
        # Display embeddings status
        st.markdown("#### Embeddings")
        if status['embeddings']['status'] == 'processing':
            st.progress(50, "⏳ Processing...")
        elif status['embeddings']['status'] == 'complete':
            st.progress(100, "✅ Complete")
        elif status['embeddings']['status'] == 'error':
            st.error("❌ Failed")
        else:
            st.progress(0, "🔄 Waiting...")

def main():
    """Main application function."""
    # Create sidebar with purple styling
    sidebar = create_sidebar_section()
    
    # Sidebar for video URL input
    with sidebar:
        st.markdown("#### 📹 Video Source")
        url_input = st.text_input(
            "YouTube URL",
            key="url_input",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter a YouTube URL to analyze"
        )
        
        st.markdown("#### 🔍 Retrieval Method")
        retrieval_options = RetrievalFactory.get_method_names()
        selected_method = st.selectbox(
            "Select Method",
            options=retrieval_options,
            format_func=lambda x: x,
            help="Choose which retrieval method to use for searching the video content"
        )
        
        # Map the selected method name back to the enum
        for method in RetrievalMethod:
            if method.value == selected_method:
                st.session_state.retrieval_method = method
                break
        
        # Add a load button with animation
        load_col1, load_col2 = st.columns(2)
        with load_col1:
            load_video = st.button("🎬 Load Video", type="primary", use_container_width=True, on_click=process_url_button_click)
        
        with load_col2:
            clear_chat = st.button("🧹 Clear Chat", use_container_width=True, on_click=clear_chat_history)
        
        # Show processing status
        display_processing_status()
        
        # Add information and instructions
        with st.expander("ℹ️ Instructions", expanded=False):
            st.markdown("""
            1. Paste a YouTube URL in the text box
            2. Select a retrieval method
            3. Click 'Load Video' to process the video
            4. Ask questions in the chat box below
            
            **Retrieval Methods**:
            - FAISS: Fast similarity search (balanced)
            - pgvector: Database-backed search (slow but accurate)
            - TF-IDF: Term frequency search (fast for text)
            - BM25: Enhanced lexical search (precise text matching)
            """)
    
    # Main content area - chat interface
    if load_video and st.session_state.youtube_url:
        # Setup database connection if needed
        if not setup_database_if_needed():
            return
        
        # Load data for the video
        with st.status("🎬 Loading video content...") as status:
            # Step 1: Load transcript segments
            status.update(label="📝 Transcribing video...", state="running")
            if not load_segments(st.session_state.youtube_url):
                status.update(label="❌ Failed to transcribe video", state="error")
                return
            
            # Step 2: Extract keyframes
            status.update(label="🎞️ Extracting key frames...", state="running")
            if not load_keyframes(st.session_state.youtube_url):
                status.update(label="⚠️ Warning: Failed to extract keyframes (continuing anyway)", state="running")
            
            # Step 3: Generate embeddings
            status.update(label="🔢 Generating embeddings...", state="running")
            if not generate_embeddings(st.session_state.youtube_url):
                status.update(label="❌ Failed to generate embeddings", state="error")
                return
            
            # All done, video is loaded
            st.session_state.video_loaded = True
            status.update(label="✅ Video loaded successfully!", state="complete")
    
    # Display chat interface
    display_chat_history(display_video_callback)
    
    # Display input box if video is loaded
    if st.session_state.video_loaded:
        chat_input_section(handle_user_query)
    
    # Custom JavaScript for handling timestamp clicks
    st.markdown("""
    <script>
    // Add event listener for timestamp clicks
    document.addEventListener('click', function(e) {
        // Check if the clicked element has the timestamp-link class
        if (e.target.classList.contains('timestamp-link')) {
            // Get the timestamp text
            const timestamp = e.target.textContent;
            
            // Send a custom event to trigger a click on the hidden button
            const event = new CustomEvent('streamlit:timeClick', { 
                detail: { timestamp: timestamp }
            });
            window.dispatchEvent(event);
        }
    });
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
