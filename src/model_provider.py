from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    provider = value.strip().lower()
    provider = {"anthorpic": "anthropic", "google": "gemini", "openai-compatible": "custom"}.get(provider, provider)
    if provider not in {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}:
        raise ValueError(f"Unsupported provider: {value}")
    return provider


def build_chat_model(config: ProviderConfig):
    """Import provider integrations only when live mode is requested."""
    provider = normalize_provider(config.provider)
    if provider in {"openai", "custom", "openrouter"}:
        from langchain_openai import ChatOpenAI

        base_url = config.base_url or ("https://openrouter.ai/api/v1" if provider == "openrouter" else None)
        if provider == "custom" and not base_url:
            raise ValueError("CUSTOM_BASE_URL is required for the custom provider")
        return ChatOpenAI(model=config.model_name, temperature=config.temperature,
                          api_key=config.api_key, base_url=base_url)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=config.model_name, temperature=config.temperature,
                                      google_api_key=config.api_key)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.model_name, temperature=config.temperature, api_key=config.api_key)
    from langchain_ollama import ChatOllama

    return ChatOllama(model=config.model_name, temperature=config.temperature,
                      base_url=config.base_url or "http://localhost:11434")
