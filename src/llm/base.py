"""
src/llm/base.py

The LLMClient contract. Everything downstream (RAG generate, extractor
agent, checker agent) talks to this abstract interface, not to Groq
or OpenAI directly. Swapping providers is a one-line change.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class LLMClient(ABC):
    """
    Abstract LLM client. Concrete implementations wrap a specific
    provider (Groq, OpenAI, Azure OpenAI, Anthropic, ...).
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """
        Return a free-text completion.

        temperature=0.0 by default because our downstream use cases
        (rule extraction, compliance checking) want determinism,
        not creativity.
        """
        ...

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Return a parsed JSON object. Providers that support native
        JSON mode (Groq, OpenAI) should use it; others fall back to
        prompt-engineered JSON + parsing.

        Raises ValueError if the model returns invalid JSON.
        """
        ...