from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    profile_confidence_threshold: float = 0.8


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    values = {}
    env_file = root / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.removeprefix("export ").split("=", 1)
                values[key.strip()] = value.strip().strip('"\'')
    values.update(os.environ)

    def provider(prefix: str, fallback: str = "openai") -> ProviderConfig:
        name = normalize_provider(values.get(f"{prefix}_PROVIDER", fallback))
        key_var = {"openai": "OPENAI_API_KEY", "custom": "CUSTOM_API_KEY",
                   "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
                   "openrouter": "OPENROUTER_API_KEY"}.get(name)
        url_var = {"custom": "CUSTOM_BASE_URL", "ollama": "OLLAMA_BASE_URL",
                   "openrouter": "OPENROUTER_BASE_URL"}.get(name)
        return ProviderConfig(name, values.get(f"{prefix}_MODEL", "gpt-4o-mini"),
                              float(values.get(f"{prefix}_TEMPERATURE", "0")),
                              values.get(key_var) if key_var else None,
                              values.get(url_var) if url_var else None)

    threshold = int(values.get("COMPACT_THRESHOLD_TOKENS", "1200"))
    keep = int(values.get("COMPACT_KEEP_MESSAGES", "4"))
    confidence = float(values.get("PROFILE_CONFIDENCE_THRESHOLD", "0.8"))
    if threshold <= 0 or keep < 1:
        raise ValueError("Compact threshold must be positive and keep_messages >= 1")
    if not 0 <= confidence <= 1:
        raise ValueError("Profile confidence threshold must be between 0 and 1")
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(root, root / "data", state_dir, threshold, keep,
                     provider("LLM"), provider("JUDGE", values.get("LLM_PROVIDER", "openai")), confidence)
