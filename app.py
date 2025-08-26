import streamlit as st
import os
import logging
import uuid
from langchain_core.messages import HumanMessage, AIMessage

# Existing imports
from config import Config
from chatbot.chatbot import Chatbot
from services.memory_manager import MemoryManager

# NEW: import RAG helpers
from rag.rag_utils import (
    get_pdf_txt, get_text_chunks, get_embeddings, get_conversation_chain
)

# --- Logging ---
if not os.path.exists(Config.LOG_DIR):
    os.makedirs(Config.LOG_DIR)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(Config.LOG_FILE),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Secure AI Chatbot", layout="wide")
    st.title("Secure AI Chatbot with Moderation + RAG")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    current_session_id = st.session_state.session_id

    # Init chatbot once
    if "chatbot_instance" not in st.session_state:
        st.session_state.chatbot_instance = Chatbot()

    chatbot_instance = st.session_state.chatbot_instance

    # --- Sidebar for RAG ---
    st.sidebar.header("📚 RAG Setup")
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    pdf_docs = st.sidebar.file_uploader("Upload PDFs:", accept_multiple_files=True)
    if st.sidebar.button("Process PDFs"):
        with st.spinner("Processing documents..."):
            raw_text = get_pdf_txt(pdf_docs)
            chunks = get_text_chunks(raw_text)
            embeddings = get_embeddings()
            vectorstore = FAISS.from_texts(chunks, embeddings)
            st.session_state.conversation_chain = get_conversation_chain(vectorstore)
            st.sidebar.success("PDFs processed successfully!")

    # --- Chat section ---
    st.header("Chat with the AI")
    user_input = st.text_input("You:", placeholder="Ask a safe question...", key="user_input_text")

    if st.button("Send Message", key="send_button", use_container_width=True):
        if user_input:
            with st.spinner("Processing your request..."):
                if st.session_state.conversation_chain:
                    # Get candidate answer from RAG
                    rag_response = st.session_state.conversation_chain({"question": user_input})
                    candidate_response = rag_response["answer"]

                    # Moderate RAG answer
                    response = chatbot_instance.get_response(candidate_response, current_session_id)
                else:
                    # Fallback: normal chatbot
                    response = chatbot_instance.get_response(user_input, current_session_id)

                st.success(f"**AI:** {response}")
        else:
            st.warning("Please enter a message before sending.")

    # --- Conversation history ---
    st.subheader("Conversation History")
    current_memory_messages = MemoryManager.get_session_history(current_session_id).messages
    if not current_memory_messages:
        st.write("Start chatting to see your conversation history here!")
    else:
        for i, message in enumerate(reversed(current_memory_messages)):
            if isinstance(message, HumanMessage):
                st.info(f"**You:** {message.content}")
            elif isinstance(message, AIMessage):
                if message.content.startswith("🚫") or message.content.startswith("⚠️"):
                    st.error(f"**AI:** {message.content}")
                else:
                    st.success(f"**AI:** {message.content}")
            if i < len(current_memory_messages) - 1:
                st.markdown("---")

    st.caption("Developed with LangChain and Streamlit")

if __name__ == "__main__":
    main()
