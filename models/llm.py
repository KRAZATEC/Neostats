"""
NeoStats AI Chatbot — LLM Models Module
Supports OpenAI, Groq, and Google Gemini with a unified interface.
"""

import logging
from abc import ABC, abstractmethod
from typing import Generator, Optional

logger = logging.getLogger(__name__)


# ─── Abstract Base ─────────────────────────────────────────────────────────────

class BaseLLM(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, model: str, temperature: float, max_tokens: int):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def chat(self, messages: list[dict], system_prompt: str = "") -> str:
        """Return full response as a string."""
        pass

    @abstractmethod
    def stream(self, messages: list[dict], system_prompt: str = "") -> Generator[str, None, None]:
        """Yield response tokens one by one."""
        pass

    def _build_messages(self, messages: list[dict], system_prompt: str) -> list[dict]:
        """Prepend system prompt to messages list."""
        full = []
        if system_prompt:
            full.append({"role": "system", "content": system_prompt})
        full.extend(messages)
        return full


# ─── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAILLM(BaseLLM):
    """OpenAI GPT models wrapper."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 temperature: float = 0.4, max_tokens: int = 1000):
        super().__init__(model, temperature, max_tokens)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"OpenAI init failed: {e}")
            raise

    def chat(self, messages: list[dict], system_prompt: str = "") -> str:
        try:
            full_messages = self._build_messages(messages, system_prompt)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise

    def stream(self, messages: list[dict], system_prompt: str = "") -> Generator[str, None, None]:
        try:
            full_messages = self._build_messages(messages, system_prompt)
            with self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            ) as response:
                for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
        except Exception as e:
            logger.error(f"OpenAI stream error: {e}")
            raise


# ─── Groq ──────────────────────────────────────────────────────────────────────

class GroqLLM(BaseLLM):
    """Groq ultra-fast inference wrapper."""

    def __init__(self, api_key: str, model: str = "llama3-8b-8192",
                 temperature: float = 0.4, max_tokens: int = 1000):
        super().__init__(model, temperature, max_tokens)
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")
        except Exception as e:
            logger.error(f"Groq init failed: {e}")
            raise

    def chat(self, messages: list[dict], system_prompt: str = "") -> str:
        try:
            full_messages = self._build_messages(messages, system_prompt)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Groq chat error: {e}")
            raise

    def stream(self, messages: list[dict], system_prompt: str = "") -> Generator[str, None, None]:
        try:
            full_messages = self._build_messages(messages, system_prompt)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            logger.error(f"Groq stream error: {e}")
            raise


# ─── Gemini ────────────────────────────────────────────────────────────────────

class GeminiLLM(BaseLLM):
    """Google Gemini models wrapper."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash",
                 temperature: float = 0.4, max_tokens: int = 1000):
        super().__init__(model, temperature, max_tokens)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.genai = genai
            gen_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            self.client = genai.GenerativeModel(model, generation_config=gen_config)
        except ImportError:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Gemini init failed: {e}")
            raise

    def _to_gemini_format(self, messages: list[dict], system_prompt: str) -> tuple[str, list]:
        """Convert OpenAI-style messages to Gemini format."""
        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})
        last_msg = messages[-1]["content"] if messages else ""
        if system_prompt:
            last_msg = f"{system_prompt}\n\n{last_msg}"
        return last_msg, history

    def chat(self, messages: list[dict], system_prompt: str = "") -> str:
        try:
            last_msg, history = self._to_gemini_format(messages, system_prompt)
            convo = self.client.start_chat(history=history)
            response = convo.send_message(last_msg)
            return response.text or ""
        except Exception as e:
            logger.error(f"Gemini chat error: {e}")
            raise

    def stream(self, messages: list[dict], system_prompt: str = "") -> Generator[str, None, None]:
        try:
            last_msg, history = self._to_gemini_format(messages, system_prompt)
            convo = self.client.start_chat(history=history)
            response = convo.send_message(last_msg, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini stream error: {e}")
            raise


# ─── Factory ───────────────────────────────────────────────────────────────────

def get_llm(provider: str, model: str, temperature: float,
            max_tokens: int, api_key: str) -> BaseLLM:
    """
    Factory function to instantiate the correct LLM provider.

    Args:
        provider: One of 'OpenAI', 'Groq', 'Gemini'
        model: Model name string
        temperature: Sampling temperature
        max_tokens: Max output tokens
        api_key: Provider API key

    Returns:
        Instantiated BaseLLM subclass
    """
    providers = {
        "OpenAI": OpenAILLM,
        "Groq": GroqLLM,
        "Gemini": GeminiLLM,
    }
    if provider not in providers:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(providers.keys())}")
    try:
        return providers[provider](
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error(f"Failed to create {provider} LLM: {e}")
        raise
