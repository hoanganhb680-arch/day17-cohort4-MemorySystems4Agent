from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        conversations = json.load(source)
    if not isinstance(conversations, list):
        raise ValueError("Dataset must contain a list of conversations")
    return conversations


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    return sum(fact.casefold() in answer.casefold() for fact in expected) / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    if not answer.strip() or "chưa có thông tin" in answer.lower():
        return 0.0
    return recall_points(answer, expected)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    user_ids = {item["user_id"] for item in conversations}
    before = sum(agent.memory_file_size(user_id) for user_id in user_ids) if isinstance(agent, AdvancedAgent) else 0
    thread_ids = []
    recall_scores = []
    quality_scores = []
    for conversation in conversations:
        thread_id = f"{conversation['id']}-turns"
        thread_ids.append(thread_id)
        for turn in conversation["turns"]:
            agent.reply(conversation["user_id"], thread_id, turn)
        for number, question in enumerate(conversation["recall_questions"]):
            recall_id = f"{conversation['id']}-recall-{number}"
            thread_ids.append(recall_id)
            answer = agent.reply(conversation["user_id"], recall_id, question["question"])["response"]
            recall_scores.append(recall_points(answer, question["expected_contains"]))
            quality_scores.append(heuristic_quality(answer, question["expected_contains"]))
    after = sum(agent.memory_file_size(user_id) for user_id in user_ids) if isinstance(agent, AdvancedAgent) else 0
    return BenchmarkRow(agent_name, sum(agent.token_usage(thread_id) for thread_id in thread_ids),
                        sum(agent.prompt_token_usage(thread_id) for thread_id in thread_ids),
                        sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
                        sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
                        after - before, sum(agent.compaction_count(thread_id) for thread_id in thread_ids))


def format_rows(rows: list[BenchmarkRow]) -> str:
    lines = ["| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row.agent_name} | {row.agent_tokens_only} | {row.prompt_tokens_processed} | "
                     f"{row.recall_score:.1%} | {row.response_quality:.1%} | "
                     f"{row.memory_growth_bytes} | {row.compactions} |")
    return "\n".join(lines)


def main() -> None:
    config = load_config()
    for title, filename in (("Standard Benchmark", "conversations.json"),
                            ("Long-Context Stress Benchmark", "advanced_long_context.json")):
        conversations = load_conversations(config.data_dir / filename)
        with tempfile.TemporaryDirectory() as state_dir:
            isolated = replace(config, state_dir=Path(state_dir))
            rows = [run_agent_benchmark("Baseline", BaselineAgent(isolated, force_offline=True), conversations, isolated),
                    run_agent_benchmark("Advanced", AdvancedAgent(isolated, force_offline=True), conversations, isolated)]
        print(f"\n## {title}\n{format_rows(rows)}")


if __name__ == "__main__":
    main()
