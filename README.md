# Secure AI Chatbot with Moderation + RAG

## 🚀 Project Overview
This project integrates **Generative AI safety and retrieval techniques** into a unified system:

- **Retrieval-Augmented Generation (RAG):** Enables users to upload PDFs and query them intelligently using LangChain, FAISS, and OpenAI GPT models.  
- **AI Moderation Layer:** Detects and blocks unsafe or policy-violating interactions using rule-based filters (hate speech, PII, jailbreaks) and LLM-based moderation.  
- **Conversation Memory:** Maintains coherent multi-turn conversations through LangChain’s `ConversationBufferMemory`.  
- **Analytics Dashboard:** Provides real-time visibility into moderation effectiveness using Streamlit and Plotly.  

This project demonstrates how to combine **safety, grounding, and observability**—three pillars of building responsible GenAI applications.

---

## 📚 Concepts Demonstrated
This project showcases practical command of:

- **Prompt Moderation & Safety Engineering**  
  Implemented hybrid moderation (regex + LLM-based) to safeguard against harmful, sensitive, or adversarial inputs.  

- **Retrieval-Augmented Generation (RAG)**  
  Built a PDF ingestion and chunking pipeline with embeddings + FAISS vector search for grounded, context-aware responses.  

- **Conversation Memory Architectures**  
  Designed session-aware memory using `ConversationBufferMemory`, enabling multi-turn dialogue with historical awareness.  

- **LLMOps & Observability**  
  Implemented logging and a monitoring dashboard to track input blocks, moderation categories, and blocked outputs for accountability.  

- **Modular AI System Design**  
  Architected the system into clearly separated modules: chatbot, moderation, retrieval, services, and analytics.  

---

## 🌟 Key Features
- **End-to-End Safe Chatbot:** Moderation → Retrieval → Memory → Response → Logging.  
- **Grounded Answers:** Upload PDFs to ask context-specific questions with reduced hallucinations.  
- **Session-Aware Conversations:** Chat history maintained separately per browser session.  
- **Analytics & Monitoring:** Dashboard visualizes moderation events (hate speech, PII, jailbreak attempts, disallowed outputs).  
- **Extensible Design:** Modular architecture allows for easy enhancements and integration.  

---

## 🛠️ Technologies Used
- **Frontend:** Streamlit (chat UI + analytics dashboard)  
- **LLM Framework:** LangChain (RAG, chains, memory)  
- **Vector Store:** FAISS  
- **Models:** OpenAI GPT (chat + moderation)  
- **Visualization:** Plotly (metrics charts)  
- **Logging:** Python logging  

---

## 📁 Project Structure
    Chatbot_Moderation_RAG/
    │── app.py # Unified chat interface (RAG + moderation)
    │── dashboard_app.py # Moderation metrics dashboard
    │── config.py # Configuration (API keys, model names, thresholds)
    │── requirements.txt
    │── README.md
    ├── chatbot/
    │ └── chatbot.py # Moderated chatbot orchestration
    ├── moderation/
    │ └── moderator.py # Rule-based + LLM moderation logic
    ├── rag/
    │ └── rag_utils.py # PDF processing and RAG chain setup
    ├── services/
    │ ├── llm_service.py # LLM instantiation layer
    │ └── memory_manager.py # Session-based memory handling
    ├── logs/
    │ └── moderation.log # Captures all user/AI interactions with moderation
    └── tests/
    └── test_moderation.py # Unit tests for moderation logic
