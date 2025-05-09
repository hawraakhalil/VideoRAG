# VideoRAG - Video Retrieval Augmented Generation

VideoRAG is a multimodal question answering system for video content. It combines text transcription, keyframe extraction, and semantic search to allow users to ask natural language questions about the content of YouTube videos and get accurate, timestamped answers.

## Features

- **YouTube Video Processing**: Automatically downloads videos, transcribes audio, and extracts keyframes.
- **Multi-modal Search**: Uses both text and image embeddings to find the most relevant content.
- **Interactive UI**: Chat interface for asking questions about the video content.
- **Timestamped Results**: Results are linked to specific timestamps in the video with confidence scores.

## Architecture

The system consists of the following main components:

1. **Data Processing Pipeline**:
   - Video download and preprocessing
   - Whisper-based transcription
   - Keyframe extraction

2. **Embedding Generation**:
   - Text embeddings for transcript segments
   - Image embeddings for keyframes

3. **Retrieval System**:
   - Multiple vector database options:
     - FAISS for in-memory similarity search
     - PostgreSQL with pgvector using IVFFLAT index
   - Fusion of text and image search results (optional)

4. **User Interface**:
   - Streamlit-based chat interface
   - Video player with timestamp-based navigation
   - Selection of different retrieval methods

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd VideoRAG
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install additional dependencies for video processing:
   - FFmpeg (required for video processing)
   - For Ubuntu/Debian: `apt-get install ffmpeg`
   - For macOS: `brew install ffmpeg`
   - For Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

5. For PostgreSQL with pgvector (optional):
   - Install PostgreSQL 12+ 
   - Install pgvector extension: [pgvector installation guide](https://github.com/pgvector/pgvector#installation)
   - Create a database for VideoRAG
   - Configure connection in `.env` file (see `docs/env_template.txt` for reference)

## Usage

1. Run the Streamlit application:
   ```
   cd VideoRAG
   streamlit run app/main.py
   ```

2. In the web interface:
   - Enter a YouTube URL in the sidebar
   - Select a retrieval method (FAISS or PostgreSQL/pgvector)
   - Click "Load Video" and wait for processing to complete
   - Start asking questions about the video content
   - View answers with timestamped video segments
  
## Project Structure

- `app/`: The Streamlit web application
  - `main.py`: Main application entry point
  - `chat_interface.py`: Chat UI components
  - `video_player.py`: Video playback functionality
  - `style.css`: Custom styling

- `models/`: ML model implementations
  - `whisper_transcriber.py`: Audio transcription using Whisper
  - `keyframe_extractor.py`: Video keyframe extraction
  - `text_embedding.py`: Text embedding generation
  - `image_embedding.py`: Image embedding generation

- `retrieval/`: Search and retrieval systems
  - `faiss_search.py`: Vector similarity search with FAISS (L2 distance)
  - `pgvector_search.py`: PostgreSQL/pgvector integration with IVFFLAT index for scalable vector search
  - `bm25_search.py`: Probabilistic lexical search using BM25 (Okapi BM25 ranking with tokenized transcripts)
  - `tfidf_search.py`: Lexical search using TF-IDF vectors and cosine similarity
  - `retrieval_factory.py`: Factory pattern to dynamically switch between FAISS, pgvector, BM25, and TF-IDF methods

- `data/`: Data storage
  - `raw_video/`: Downloaded video files
  - `transcripts/`: Transcript segments
  - `keyframes/`: Extracted video keyframes
  - `embeddings/`: Generated embeddings
  - `gold_set.json`: Includes 15 answerable questions from the video, and 15 unanswerable

## License

This project is licensed under the MIT license.
