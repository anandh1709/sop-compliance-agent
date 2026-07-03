"""
src/config.py

Central place to load environment variables. Everything else in the
codebase imports from here rather than calling os.getenv directly,
so if we ever move to a secrets manager (Azure Key Vault, AWS Secrets
Manager) we swap the implementation here, not in 20 places.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file).
load_dotenv(Path(__file__).parent.parent / ".env")


# --- Groq (LLM) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Azure AI Search ---
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "sop-chunks")


def require(name: str, value: str | None) -> str:
    """
    Raise a clear error if a required env var is missing.
    Used at the entry points of Azure/Groq code paths so we fail fast
    with a helpful message rather than deep in an HTTP client.
    """
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            f"Check your .env file at the project root."
        )
    return value