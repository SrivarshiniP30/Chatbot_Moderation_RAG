import streamlit as st
import logging
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory # For type hinting

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages LangChain's ConversationBufferMemory instances, storing them
    within Streamlit's session_state for persistence across app reruns.
    Each Streamlit session (i.e., browser tab) gets its own unique conversation memory.
    """
    @staticmethod
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        """
        Retrieves or creates a ConversationBufferMemory instance for a given session ID.
        This function is designed to be passed to LangChain's RunnableWithMessageHistory.
        It stores the memory objects within Streamlit's session_state.
        """
        # Initialize the 'langchain_memory_store' dictionary within Streamlit's session_state
        # if it doesn't already exist for the current Streamlit session.
        if "langchain_memory_store" not in st.session_state:
            st.session_state.langchain_memory_store = {}
            logger.info("Initialized Streamlit's session_state.langchain_memory_store.")

        # If a ConversationBufferMemory instance for the current session_id doesn't exist
        # in our session-state-managed store, create a new one.
        if session_id not in st.session_state.langchain_memory_store:
            logger.info(f"Creating new ConversationBufferMemory for session ID: {session_id}")
            st.session_state.langchain_memory_store[session_id] = ConversationBufferMemory(
                return_messages=True, # Important: memory returns actual message objects (HumanMessage, AIMessage)
                memory_key="history" # This key should match the 'variable_name' in MessagesPlaceholder in your prompt
            )
        # Return the underlying ChatMessageHistory object from the ConversationBufferMemory instance.
        # This object directly holds the list of chat messages.
        return st.session_state.langchain_memory_store[session_id].chat_memory

