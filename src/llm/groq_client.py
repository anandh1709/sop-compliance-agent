"""
src/llm/groq_client.py

Groq-backed LLM client. Chosen for speed (sub-second inference on
llama-3.1-8b) and generous free tier during development.

Swap to OpenAIClient / AzureOpenAIClient for prod without touching
any calling code — that's what the LLMClient abstraction buys us.
"""
import json
import time
from typing import Optional, Dict, Any

from groq import Groq, APIError

from src.config import GROQ_API_KEY, require
from src.llm.base import LLMClient


class GroqClient(LLMClient):
    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        max_retries: int = 1,
    ):
        """
        model: default llama-3.1-8b-instant — cheap and fast, sufficient
               for extraction and compliance-check tasks. Bump to
               llama-3.3-70b-versatile if quality is insufficient.

        max_retries: one retry on transient errors (rate limits,
                     5xx). Keeps demos from failing on flukes.
        """
        self._api_key = require("GROQ_API_KEY", GROQ_API_KEY)
        self._client = Groq(api_key=self._api_key)
        self.model = model
        self.max_retries = max_retries

    def _messages(self, prompt: str, system: Optional[str]):
        """Assemble the OpenAI-style messages array."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _call_with_retry(self, **kwargs) -> str:
        """
        One-shot retry on transient errors. Not exponential backoff
        because for a tonight-demo, one retry after 2s is enough.
        Prod would use tenacity with jittered exponential backoff.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except APIError as e:
                last_exc = e
                # Only retry on server-side / rate-limit errors, not on
                # 4xx bad requests which won't get better on retry.
                if getattr(e, "status_code", None) in (429, 500, 502, 503, 504):
                    if attempt < self.max_retries:
                        time.sleep(2)
                        continue
                raise
        # Shouldn't reach here, but keep mypy happy.
        raise last_exc  # type: ignore[misc]

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        return self._call_with_retry(
            model=self.model,
            messages=self._messages(prompt, system),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Uses Groq's native JSON mode. When response_format={"type":
        "json_object"} is set, the model is constrained to output
        valid JSON — no regex extraction, no parsing failures on
        markdown fences.

        NOTE: Groq (like OpenAI) requires the word 'json' to appear
        somewhere in the prompt or system message when using JSON
        mode. Otherwise you get a 400 error. Callers should include
        an instruction like "Respond with a JSON object..." in the
        prompt.
        """
        raw = self._call_with_retry(
            model=self.model,
            messages=self._messages(prompt, system),
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            # This shouldn't happen with JSON mode enabled, but if it
            # does, surface the raw output so we can debug.
            raise ValueError(
                f"Model returned invalid JSON despite JSON mode. "
                f"Raw response: {raw[:500]}"
            ) from e


if __name__ == "__main__":
    # Smoke test: prove both generate() and generate_json() work
    # against Groq's live API. Uses a trivial prompt so we don't
    # burn tokens.
    client = GroqClient()
    print(f"Model: {client.model}\n")

    print("--- generate() ---")
    text = client.generate(
        prompt="In one sentence, what is an SOP in a clinical trial context?",
        system="You are a concise regulatory affairs expert.",
    )
    print(text, "\n")

    print("--- generate_json() ---")
    obj = client.generate_json(
        prompt=(
            "Extract the timing rule from this text and respond as a "
            "JSON object with fields 'action' and 'deadline':\n\n"
            "'Adverse events must be reported to the sponsor within "
            "24 hours of the investigator becoming aware of them.'"
        ),
    )
    print(json.dumps(obj, indent=2))