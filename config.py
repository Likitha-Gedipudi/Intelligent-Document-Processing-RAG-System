"""
Configuration for Banking Document RAG System
All paths and settings in one place
"""
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chroma_db"
SAMPLE_DOCS_DIR = BASE_DIR / "sample_docs"

# Database
DATABASE_PATH = DATA_DIR / "banking.db"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model (free, runs locally)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Ollama settings (local LLM)
OLLAMA_MODEL = "llama3:8b"  # or "phi3:mini" for lower RAM usage
OLLAMA_HOST = "http://localhost:11434"

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Document types
DOCUMENT_TYPES = [
    "loan_application",
    "kyc_document", 
    "bank_statement",
    "salary_slip",
    "other"
]

# Supported file extensions
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx"]
