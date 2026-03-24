import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    CHROMA_HOST = os.getenv("CHROMA_HOST", "http://chromadb:8000")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")
    CHATBOT_HOST = os.getenv("CHATBOT_HOST", "0.0.0.0")
    CHATBOT_PORT = int(os.getenv("CHATBOT_PORT", 8000))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DOCUMENTS_PATH = "/app/documents"
