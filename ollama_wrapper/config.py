import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Base URL for Ollama API
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

    # API endpoints
    GENERATE_ENDPOINT = "/api/generate"
    CHAT_ENDPOINT = "/api/chat"
    CREATE_MODEL_ENDPOINT = "/api/create"
    LIST_MODELS_ENDPOINT = "/api/tags"
    SHOW_MODEL_ENDPOINT = "/api/show"
    COPY_MODEL_ENDPOINT = "/api/copy"
    DELETE_MODEL_ENDPOINT = "/api/delete"
    PULL_MODEL_ENDPOINT = "/api/pull"
    PUSH_MODEL_ENDPOINT = "/api/push"
    EMBEDDINGS_ENDPOINT = "/api/embeddings"
    RUNNING_MODELS_ENDPOINT = "/api/running"
    VERSION_ENDPOINT = "/api/version"

    # Request defaults
    DEFAULT_TIMEOUT = 60
    DEFAULT_HEADERS = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"