from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import answer_recall, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)
        state = self.sessions.setdefault(thread_id, SessionState())
        state.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in state.messages)
        result = self.langchain_agent.invoke(state.messages)
        response = str(result.content)
        state.messages.append({"role": "assistant", "content": response})
        tokens = estimate_tokens(response)
        state.token_usage += tokens
        state.prompt_tokens_processed += prompt_tokens
        return {"response": response, "tokens": tokens, "prompt_tokens": prompt_tokens}

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        state = self.sessions.setdefault(thread_id, SessionState())
        facts: dict[str, str] = {}
        for item in state.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        facts.update(extract_profile_updates(message))
        state.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in state.messages)
        response = answer_recall(message, facts)
        state.messages.append({"role": "assistant", "content": response})
        tokens = estimate_tokens(response)
        state.token_usage += tokens
        state.prompt_tokens_processed += prompt_tokens
        return {"response": response, "tokens": tokens, "prompt_tokens": prompt_tokens}

    def _maybe_build_langchain_agent(self):
        if self.force_offline or (not self.config.model.api_key and self.config.model.provider != "ollama"):
            return None
        return build_chat_model(self.config.model)
