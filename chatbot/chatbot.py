# chatbot_moderation_project/chatbot/chatbot.py
import logging
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage # Import these to manually add to memory

from moderation.moderator import ChatbotModerator
from services.llm_service import LLMService
from services.memory_manager import MemoryManager
from config import Config

logger = logging.getLogger(__name__)

class Chatbot:
    """
    Main chatbot class, integrating moderation and conversation memory.
    """
    def __init__(self):
        # Initialize LLM for chatbot responses using LLMService
        self.llm = LLMService.get_chat_llm()
        if self.llm is None:
            st.error("Main Chat LLM could not be initialized. Please ensure OPENAI_API_KEY is set correctly.")
            st.stop()

        self.moderator = ChatbotModerator()

        # Define the main chat prompt with a messages placeholder for history
        self.main_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and friendly assistant. Respond concisely to the user's query."),
            MessagesPlaceholder(variable_name="history"), # Matches history_messages_key in RunnableWithMessageHistory
            ("user", "{query}")
        ])

        # Create the core chain (prompt | llm | parser)
        core_chain = self.main_chat_prompt | self.llm | StrOutputParser()

        # Wrap the core chain with RunnableWithMessageHistory for memory management
        self.chain_with_history = RunnableWithMessageHistory(
            core_chain,
            MemoryManager.get_session_history, # Function to retrieve/create memory per session
            input_messages_key="query", # Key in the input dict that is the current user query
            history_messages_key="history" # Key in the prompt template to populate with history
        )

    def get_response(self, user_input: str, session_id: str) -> str:
        """
        Gets a response from the LLM after moderating the input, integrating LangChain's memory.
        Also moderates the LLM's generated output and ensures blocked interactions are logged to memory.
        """
        # Get the chat history manager for the current session
        session_history = MemoryManager.get_session_history(session_id)

        # Step 1: Moderate the user's input before sending to the main LLM
        is_allowed_input, input_moderation_reason = self.moderator.moderate_input(user_input)
        moderation_response_to_user = ""

        if not is_allowed_input:
            moderation_response_to_user = f"🚫 Your input was blocked: {input_moderation_reason}"
            logger.info(f"User input was blocked by moderation: {input_moderation_reason} - '{user_input}' (Session: {session_id})")
            # Manually add the user's input and the moderation response to memory
            session_history.add_user_message(user_input)
            session_history.add_ai_message(moderation_response_to_user)
            return moderation_response_to_user # Return explicit message for user

        logger.info(f"User input accepted; sending to main LLM: '{user_input}' (Session: {session_id})")

        llm_response = "I'm sorry, I encountered an issue while processing your request. Please try again." # Default error message

        try:
            # Step 2: Invoke the main chatbot chain with the user's query and session ID.
            # RunnableWithMessageHistory handles loading/saving history automatically for *successful* turns.
            llm_response = self.chain_with_history.invoke(
                {"query": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            logger.info(f"Main LLM Raw Response: '{llm_response}' for input: '{user_input}' (Session: {session_id})")

            # Step 3: Moderate the LLM's generated response
            is_allowed_output, output_moderation_reason = self.moderator.moderate_input(llm_response)

            if not is_allowed_output:
                moderation_response_to_user = f"⚠️ My response was blocked due to policy violation: {output_moderation_reason}"
                logger.warning(f"LLM output blocked by moderation: {output_moderation_reason} - Raw Response: '{llm_response}' (Session: {session_id})")
                # When LLM output is blocked *after* it's generated, RunnableWithMessageHistory has already added
                # the user's input and the LLM's raw (potentially harmful) response to memory.
                # To ensure the UI shows our polite blocked message and that message is also in memory,
                # we need to remove the last (harmful) AI message and add our custom one.
                if session_history.messages and isinstance(session_history.messages[-1], AIMessage):
                    session_history.messages.pop() # Remove the last AI message
                session_history.add_ai_message(moderation_response_to_user)
                return moderation_response_to_user
            else:
                logger.info(f"Main LLM Response passed output moderation: '{llm_response}' (Session: {session_id})")
                return llm_response

        except Exception as e:
            logger.error(f"Error getting response from main LLM for session {session_id}: {e}")
            # If main LLM call fails, manually add the user input and error message to memory
            # The user message would have been added by the chain.invoke call if it was successful up to that point.
            # We ensure the AI's response is the error message.
            if session_history.messages and isinstance(session_history.messages[-1], AIMessage):
                 session_history.messages.pop() # Remove any partial/failed AI response
            session_history.add_ai_message(llm_response) # Add the error message as AI's response
            return llm_response
