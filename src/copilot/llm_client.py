"""
llm_client.py – Provider-independent LLM interface for the AI Security Copilot.

Supports:
    - OpenAI (gpt-4o-mini default)
    - Azure OpenAI
    - Ollama (local)
    - TemplateLLMClient (zero-dependency fallback — always works)

Auto-detection order:
    1. COPILOT_PROVIDER env var (force a provider)
    2. OPENAI_API_KEY set → OpenAI
    3. AZURE_OPENAI_KEY set → Azure
    4. Ollama reachable at OLLAMA_HOST → Ollama
    5. TemplateLLMClient (default)
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("Copilot.LLMClient")

# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Generate a response.

        Parameters
        ----------
        prompt : str   User / assistant message.
        system : str   System instruction.

        Returns
        -------
        str  Model response text.
        """

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIClient(LLMClient):
    """OpenAI chat completion client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai to use the OpenAI provider.")

        self._openai = openai
        self._client = openai.OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        logger.info("OpenAI client initialised (model=%s).", model)

    def generate(self, prompt: str, system: str = "") -> str:
        t0 = time.time()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        text = response.choices[0].message.content or ""
        logger.debug("OpenAI response in %.2fs (%d chars).", time.time() - t0, len(text))
        return text


# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------

class AzureOpenAIClient(LLMClient):
    """Azure OpenAI chat completion client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: str = "2024-02-01",
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai to use the Azure OpenAI provider.")

        self._client = openai.AzureOpenAI(
            api_key=api_key or os.environ.get("AZURE_OPENAI_KEY", ""),
            azure_endpoint=endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_version=api_version,
        )
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        self.max_tokens = max_tokens
        self.temperature = temperature
        logger.info("Azure OpenAI client initialised (deployment=%s).", self.deployment)

    def generate(self, prompt: str, system: str = "") -> str:
        t0 = time.time()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        text = response.choices[0].message.content or ""
        logger.debug("Azure OpenAI response in %.2fs.", time.time() - t0)
        return text


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------

class OllamaClient(LLMClient):
    """Ollama local LLM client (uses httpx)."""

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        try:
            import httpx
            self._httpx = httpx
        except ImportError:
            raise ImportError("pip install httpx to use the Ollama provider.")

        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.timeout = timeout
        logger.info("Ollama client initialised (host=%s, model=%s).", self.host, self.model)

    def generate(self, prompt: str, system: str = "") -> str:
        t0 = time.time()
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        with self._httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.host}/api/generate", json=payload)
            resp.raise_for_status()
            text = resp.json().get("response", "")

        logger.debug("Ollama response in %.2fs (%d chars).", time.time() - t0, len(text))
        return text

    @classmethod
    def is_reachable(cls, host: str = "http://localhost:11434") -> bool:
        """Check whether Ollama is running at *host*."""
        try:
            import httpx
            with httpx.Client(timeout=2) as c:
                r = c.get(f"{host.rstrip('/')}/api/tags")
                return r.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Template fallback (zero dependencies)
# ---------------------------------------------------------------------------

class TemplateLLMClient(LLMClient):
    """High-quality template-based fallback — works with no API key.

    Produces structured, professional SOC report sections by combining
    the rich context already assembled in the IncidentContext object
    with pre-written professional templates.
    """

    def generate(self, prompt: str, system: str = "") -> str:
        # The prompt already contains the full structured context.
        # Extract the context block and return it with minimal formatting.
        # The report_generator and summarizer handle all real formatting —
        # this fallback is only called for freeform analyst Q&A.
        return (
            "**[Template Mode — No LLM Configured]**\n\n"
            "The structured incident report and recommendations below are generated "
            "from the Cyber Cage pipeline outputs (SHAP evidence, MITRE mapping, "
            "behavioural indicators) without an external LLM.\n\n"
            "To enable AI-enhanced narrative generation, set the OPENAI_API_KEY "
            "environment variable, or start Ollama locally."
        )


# ---------------------------------------------------------------------------
# Auto-detect and factory
# ---------------------------------------------------------------------------

def create_client(provider: Optional[str] = None) -> LLMClient:
    """Detect the best available LLM provider and return a configured client.

    Parameters
    ----------
    provider : str, optional
        Force a specific provider: ``"openai"``, ``"azure"``, ``"ollama"``,
        or ``"template"``.  If *None*, auto-detects from environment.

    Returns
    -------
    LLMClient
    """
    forced = provider or os.environ.get("COPILOT_PROVIDER", "").lower()

    if forced == "openai" or (not forced and os.environ.get("OPENAI_API_KEY")):
        try:
            client = OpenAIClient(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            )
            logger.info("LLM provider: OpenAI (%s)", client.model)
            return client
        except Exception as e:
            logger.warning("OpenAI init failed: %s — falling back.", e)

    if forced == "azure" or (not forced and os.environ.get("AZURE_OPENAI_KEY")):
        try:
            client = AzureOpenAIClient()
            logger.info("LLM provider: Azure OpenAI (%s)", client.deployment)
            return client
        except Exception as e:
            logger.warning("Azure OpenAI init failed: %s — falling back.", e)

    if forced == "ollama" or (not forced and OllamaClient.is_reachable(
        os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )):
        try:
            client = OllamaClient()
            logger.info("LLM provider: Ollama (%s)", client.model)
            return client
        except Exception as e:
            logger.warning("Ollama init failed: %s — falling back.", e)

    logger.info(
        "LLM provider: Template (no API key detected — full reports still generated from structured context)."
    )
    return TemplateLLMClient()
