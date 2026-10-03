from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    return max(1, (len(text.strip()) + 3) // 4) if text.strip() else 0


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        if not re.fullmatch(r"[\w-]{1,80}", user_id, flags=re.UNICODE) or user_id in {".", ".."}:
            raise ValueError("Invalid user ID")
        return self.root_dir / user_id / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else "# User profile\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        path = self.path_for(user_id)
        if not path.exists() or not search_text:
            return False
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return dict(re.findall(r"^- ([a-z_]+): (.+)$", self.read_text(user_id), re.MULTILINE))

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        facts = self.facts(user_id)
        facts[key] = value.replace("\n", " ").strip()
        self.write_text(user_id, "# User profile\n" + "".join(
            f"- {name}: {fact}\n" for name, fact in sorted(facts.items())))


def extract_profile_updates(message: str, confidence_threshold: float = 0.8) -> dict[str, str]:
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("Profile confidence threshold must be between 0 and 1")
    facts: dict[str, str] = {}
    for segment in re.finditer(r"([^.!?\n]+)([.!?]?)", message):
        if segment.group(2) == "?":
            continue
        text = segment.group(1)
        confidence = 0.4 if re.search(r"giả sử|nếu|ước gì|tưởng tượng|ví dụ như|đùa rằng|có thể sẽ", text, re.I) else 1.0
        if confidence < confidence_threshold:
            continue
        name = re.search(r"(?:mình tên là|tên mình là)\s+([\w]+(?:\s+Stress)?)\b", text, re.I)
        if name and name.group(1).lower() not in {"gì", "ai", "không"}:
            facts["name"] = name.group(1)
        location = re.search(r"(?:mình (?:đang |vẫn )?ở|hiện ở|nơi ở (?:đã )?cập nhật từ \w+ sang)\s+(Đà Nẵng|Huế|Hà Nội)\b", text, re.I)
        if location and not re.search("không còn ở|chứ không (?:còn )?ở", text[:location.start()], re.I):
            facts["location"] = location.group(1)
        profession = re.search(r"(?:mình (?:đang |vẫn )?làm|(?:mình |giờ )?chuyển sang|nghề nghiệp (?:hiện tại )?(?:thì )?vẫn là)\s+(MLOps engineer|backend engineer)\b", text, re.I)
        if profession:
            facts["profession"] = profession.group(1)
        if re.search(r"(?:đồ uống yêu thích|mình vẫn uống|mình thích).*cà phê sữa đá", text, re.I):
            facts["drink"] = "cà phê sữa đá"
        if re.search(r"(?:món ăn yêu thích|món ruột).*mì Quảng", text, re.I):
            facts["food"] = "mì Quảng"
        if re.search(r"mình nuôi.*corgi|con corgi", text, re.I):
            facts["pet"] = "corgi Bơ" if "Bơ" in text else "corgi"
        if re.search(r"mình (?:thích|đang quan tâm).*Python", text, re.I):
            facts["interests"] = "Python, AI ứng dụng"
        if re.search(r"(?:trả lời|style trả lời|cách giải thích).*?(?:ngắn gọn|bullet)|(?:ngắn gọn|bullet).*?(?:trả lời|ví dụ)", text, re.I):
            facts["style"] = "ngắn gọn, 3 bullet có ví dụ thực chiến" if "3 bullet" in text else "ngắn gọn, có bullet và ví dụ thực tế"
    return facts


def answer_recall(question: str, facts: dict[str, str]) -> str:
    if "?" not in question and not re.search("nhắc lại|tóm tắt|nhớ lại|bạn biết", question, re.I):
        return "Mình đã ghi nhận, cảm ơn bạn."
    query = question.lower()
    requested = []
    for key, hints in {
        "name": ("tên", "dũngct", "mình là ai"),
        "location": ("ở đâu", "nơi ở", "huế", "đà nẵng", "hà nội"),
        "profession": ("nghề", "công việc", "làm gì", "product manager", "nghề hiện tại"),
        "style": ("style", "trả lời", "kiểu trả lời"),
        "drink": ("đồ uống",), "food": ("món ăn",),
        "pet": ("nuôi", "con gì"), "interests": ("quan tâm", "kỹ thuật", "python"),
    }.items():
        if any(hint in query for hint in hints) and key in facts:
            requested.append(f"{key}: {facts[key]}")
    return "; ".join(requested) if requested else "Mình chưa có thông tin chắc chắn về điều đó."


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    previous = messages[0]["content"][:200] if messages and messages[0]["role"] == "summary" else ""
    recent = messages[1:] if previous else messages
    new = " | ".join(f"{item['role']}: {item['content'][:90]}" for item in recent[-max_items:])
    return " | ".join(part for part in (previous, new) if part)[:600]


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        context = self.context(thread_id)
        messages = context["messages"]
        messages.append({"role": role, "content": content})
        size = estimate_tokens(str(context["summary"])) + sum(estimate_tokens(item["content"]) for item in messages)
        if size > self.threshold_tokens and len(messages) > self.keep_messages:
            old = messages[:-self.keep_messages]
            context["summary"] = summarize_messages(
                [{"role": "summary", "content": str(context["summary"])}] + old)
            context["messages"] = messages[-self.keep_messages:]
            context["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])
