import os
from dotenv import load_dotenv

from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    # LLM Config
    LLM_PROVIDER: str = "openai"  # openai, anthropic, local
    LLM_MODEL: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2000
    
    # VLM Config (for Perception)
    VLM_ENABLED: bool = True
    VLM_MODEL: str = os.getenv("VLM_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free")
    
    # Memory Config
    VECTOR_DB_TYPE: str = "qdrant"  # qdrant, weaviate
    VECTOR_DB_HOST: str = "localhost"
    VECTOR_DB_PORT: int = 6333
    EMBEDDER_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Perception Config
    PERCEPTION_STRATEGIES: list = [
        "text",
        "vision",
        "structured",
        "tools"
    ]
    
    # Execution Config
    MAX_RETRIES: int = 3
    EXECUTION_TIMEOUT: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()
