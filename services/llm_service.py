import logging
from langchain_openai import ChatOpenAI
from config import Config # Import Config to access LLM settings and API key

logger = logging.getLogger(__name__)

class LLMService:
    """
    A service class to centralize the initialization and management of LLM instances.
    This promotes DRY (Don't Repeat Yourself) and makes it easier to swap or configure LLMs.
    """
    _chat_llm_instance = None       # Stores the LLM instance for the main chatbot
    _moderation_llm_instance = None # Stores the LLM instance for the moderation layer

    @classmethod
    def get_chat_llm(cls):
        """
        Returns the LLM instance configured for the main chatbot.
        Initializes it if it doesn't exist to ensure singleton-like behavior.
        """
        if cls._chat_llm_instance is None:
            # Check if API key is provided before initializing LLM
            if not Config.OPENAI_API_KEY:
                logger.error("OPENAI_API_KEY is not set for chat LLM. Cannot initialize.")
                # Return None or raise an exception if LLM cannot be initialized
                return None
            cls._chat_llm_instance = ChatOpenAI(
                openai_api_key=Config.OPENAI_API_KEY,
                model_name=Config.LLM_MODEL_NAME,
                temperature=Config.LLM_TEMPERATURE_CHATBOT # Use specific temperature for chat
            )
            logger.info(f"Initialized Chat LLM: {Config.LLM_MODEL_NAME} with temperature {Config.LLM_TEMPERATURE_CHATBOT}")
        return cls._chat_llm_instance

    @classmethod
    def get_moderation_llm(cls):
        """
        Returns the LLM instance configured for the moderation layer.
        Initializes it if it doesn't exist.
        """
        if cls._moderation_llm_instance is None:
            # Check if API key is provided
            if not Config.OPENAI_API_KEY:
                logger.error("OPENAI_API_KEY is not set for moderation LLM. Cannot initialize.")
                return None
            cls._moderation_llm_instance = ChatOpenAI(
                openai_api_key=Config.OPENAI_API_KEY,
                model_name=Config.LLM_MODEL_NAME,
                temperature=Config.LLM_TEMPERATURE_MODERATION # Use specific temperature for moderation
            )
            logger.info(f"Initialized Moderation LLM: {Config.LLM_MODEL_NAME} with temperature {Config.LLM_TEMPERATURE_MODERATION}")
        return cls._moderation_llm_instance

