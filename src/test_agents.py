from __future__ import annotations

from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import load_conversations, run_agent_benchmark
from config import load_config
from memory_store import UserProfileStore, extract_profile_updates


def make_config(tmp_path: Path):
    config = load_config(tmp_path)
    config.compact_threshold_tokens = 80
    config.compact_keep_messages = 2
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.read_text("dungct") == "# User profile\n"
    assert store.file_size("dungct") == 0
    store.write_text("dungct", "# User profile\n- location: Huế\n")
    assert store.edit_text("dungct", "Huế", "Đà Nẵng")
    assert not store.edit_text("dungct", "Huế", "Hà Nội")
    store.upsert_fact("dungct", "location", "Huế")
    assert store.facts("dungct")["location"] == "Huế"
    assert store.file_size("dungct") > 0
    with pytest.raises(ValueError):
        store.path_for("../escape")


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for number in range(12):
        agent.reply("dungct", "long", f"Đây là lượt {number}: " + "ngữ cảnh dài " * 30)
    assert agent.compaction_count("long") >= 2
    assert len(agent.compact_memory.context("long")["messages"]) <= 2
    assert agent.compact_memory.context("long")["summary"]


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for agent in (baseline, advanced):
        agent.reply("dungct", "first", "Mình tên là DũngCT. Mình ở Huế.")
        assert "DũngCT" in agent.reply("dungct", "first", "Mình tên gì?")["response"]
    question = "Mình tên gì và đang ở đâu?"
    assert "DũngCT" not in baseline.reply("dungct", "second", question)["response"]
    assert "DũngCT" in advanced.reply("dungct", "second", question)["response"]
    assert "Huế" in AdvancedAgent(config, force_offline=True).reply("dungct", "third", question)["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for number in range(20):
        message = f"Lượt {number}: " + "nội dung kiểm tra dài " * 50
        baseline.reply("user", "long", message)
        advanced.reply("user", "long", message)
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
    assert advanced.compaction_count("long") > 0


def test_corrections_and_noise(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for message in ("Mình tên là DũngCT. Mình ở Huế và đang làm backend engineer.",
                    "Mình đang ở Đà Nẵng, không còn ở Huế.",
                    "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.",
                    "Hà Nội chỉ là nơi đi họp; product manager chỉ là câu đùa."):
        agent.reply("user", "first", message)
    profile = agent.profile_store.facts("user")
    assert profile["location"] == "Đà Nẵng"
    assert profile["profession"] == "MLOps engineer"
    assert extract_profile_updates("Tên mình là gì? Mình đang làm nghề gì?") == {}
    assert extract_profile_updates("Mình ở Huế? Mình tên là DũngCT?") == {}


def test_profile_confidence_threshold(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    agent.reply("user", "one", "Mình đang ở Đà Nẵng.")
    agent.reply("user", "one", "Giả sử mình ở Huế, hãy lấy ví dụ phù hợp.")
    assert agent.profile_store.facts("user")["location"] == "Đà Nẵng"
    assert "Đà Nẵng" in agent.reply("user", "two", "Mình đang ở đâu?")["response"]
    assert extract_profile_updates("Nếu mình ở Huế.") == {}
    assert extract_profile_updates("Nếu mình ở Huế.", 0.3)["location"] == "Huế"
    agent.reply("user", "one", "Mình đang ở Huế.")
    assert agent.profile_store.facts("user")["location"] == "Huế"
    with pytest.raises(ValueError):
        extract_profile_updates("Mình ở Huế.", 1.2)


def test_benchmark_stress(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    conversations = load_conversations(Path(__file__).resolve().parent.parent / "data/advanced_long_context.json")
    baseline = run_agent_benchmark("Baseline", BaselineAgent(config, True), conversations, config)
    advanced = run_agent_benchmark("Advanced", AdvancedAgent(config, True), conversations, config)
    assert advanced.recall_score > baseline.recall_score
    assert advanced.prompt_tokens_processed < baseline.prompt_tokens_processed
    assert advanced.memory_growth_bytes > 0
    assert advanced.compactions > 0
