import os

class Config:
    """
    Configuration settings for the Chatbot Moderation Project.
    Environment variables are preferred for sensitive information like API keys.
    """
    # OpenAI API Key: Fetched from environment variable for security.
    # IMPORTANT: Ensure OPENAI_API_KEY is set in your environment (e.g., export OPENAI_API_KEY="sk-...")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Large Language Model (LLM) settings
    # This model will be used by both the main chatbot and the moderation layer.
    LLM_MODEL_NAME = "gpt-4o-mini"
    LLM_TEMPERATURE_CHATBOT = 0.7   # Higher temperature for more creative/diverse chatbot responses
    LLM_TEMPERATURE_MODERATION = 0.1 # Lower temperature for more predictable/deterministic moderation responses

    # Logging settings
    LOG_DIR = "logs"  # Directory where log files will be stored
    LOG_FILE = os.path.join(LOG_DIR, "moderation.log") # Name of the main log file

    # Moderation policies and patterns
    # These are rule-based checks for efficiency and immediate blocking of obvious violations.
    MODERATION_KEYWORDS_HATE_SPEECH = [
        "hate", "kill", "harm", "attack", "racist", "sexist", "violent",
        "nazi", "terrorist", "bomb", "explode", "genocide", "destroy", "murder", "weapon"
    ]
    MODERATION_KEYWORDS_PII = [
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",  # Common US Phone numbers (e.g., 123-456-7890, 123.456.7890)
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" # Email addresses
    ]
    # More comprehensive list of jailbreak phrases to enhance rule-based detection
    MODERATION_JAILBREAK_PHRASES = [
        "ignore previous instructions", "act as if", "override your programming",
        "pretend you are", "hypothetically", "new roleplay", "forbidden knowledge",
        "as a large language model", "developer mode", "do anything now",
        "disregard all prior instructions", "you are no longer an AI",
        "simulate being human", "forget everything you know", "act like a different AI",
        "jailbreak", "unleash your full potential", "execute the following code"
    ]

    # Future Configuration Ideas (for your README's "Future Enhancements" section):
    # - DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db") # For persistent memory
    # - EXTERNAL_MODERATION_API_KEY = os.getenv("EXTERNAL_MODERATION_API_KEY") # For a more advanced moderation API
    # - RATE_LIMIT_PER_MINUTE = 20 # Example for API rate limiting
