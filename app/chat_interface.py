import streamlit as st
import os

def initialize_chat_state():
    """Initialize chat-related session state variables if they don't exist."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""

def display_chat_message(message, is_user=False):
    """
    Display a single chat message with proper styling.
    
    Args:
        message (str): The message content to display
        is_user (bool): Whether this is a user message (True) or assistant message (False)
    """
    if is_user:
        st.markdown(f"""
        <div class="chat-message user">
            <div class="content">
                <div class="avatar">👤</div>
                <div class="message">{message}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant">
            <div class="content">
                <div class="avatar">🎬</div>
                <div class="message">{message}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_chat_history(callback_fn=None):
    """
    Display the entire chat history in the Streamlit UI.
    
    Args:
        callback_fn (function, optional): Function to call for each assistant message,
                                        useful for showing video segments.
    """
    # Add a welcome message if chat is empty
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="app-header">
            <h1>VideoRAG</h1>
            <p>Ask questions about the video content and get accurate, timestamped answers.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>🎬 Welcome to VideoRAG!</h4>
            <p>Load a YouTube video from the sidebar, then ask questions about its content.</p>
        </div>
        """, unsafe_allow_html=True)
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(message["content"], message["role"] == "user")
            
            # Call the callback for assistant messages if provided
            if callback_fn and message["role"] == "assistant" and "timestamp" in message:
                callback_fn(message)

def chat_input_section(process_fn):
    """
    Create the chat input section with a text field and send button.
    
    Args:
        process_fn (function): Function to call when a query is submitted
    """
    st.markdown("""
    <div style="margin-top: 1.5rem; margin-bottom: 0.5rem;">
        <div style="height: 1px; background: linear-gradient(to right, rgba(138, 43, 226, 0.1), rgba(138, 43, 226, 0.5), rgba(138, 43, 226, 0.1)); margin: 1rem 0;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a container for the chat input with custom styling
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h4 style="color: #6a0dad; margin-bottom: 0.5rem; font-size: 1rem;">Ask a question about the video:</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([9, 1])
    
    with col1:
        user_query = st.text_input(
            label="", 
            value="", 
            key="user_input",
            placeholder="Type your question here..."
        )
    
    with col2:
        send_button = st.button("📤", use_container_width=True)
    
    # Process query when Enter is pressed or Send button is clicked
    if send_button or (user_query and user_query != st.session_state.get("last_query", "")):
        st.session_state.last_query = user_query
        process_fn(user_query)
        # Rerun to update the chat display
        st.rerun()

def add_message_to_chat(role, content, **kwargs):
    """
    Add a message to the chat history.
    
    Args:
        role (str): Either 'user' or 'assistant'
        content (str): The message content
        **kwargs: Additional data to store with the message
    """
    message = {"role": role, "content": content}
    message.update(kwargs)
    st.session_state.chat_history.append(message)

def clear_chat_history():
    """Clear the chat history."""
    st.session_state.chat_history = []
    st.session_state.last_query = ""

def create_sidebar_section():
    """Create a styled sidebar section for the app."""
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #8a2be2; font-size: 1.8rem; margin-bottom: 0.5rem;">VideoRAG</h1>
        <p style="color: #666; font-size: 0.9rem;">Video Retrieval Augmented Generation</p>
    </div>
    """, unsafe_allow_html=True)
    
    return st.sidebar.container()
