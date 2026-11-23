from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

STORE_PATH = Path(
    os.getenv("CHAT_STORE", Path("data") / "chat_history" / "chats.json")
)
_LOCK = threading.Lock()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read() -> dict:
    if not STORE_PATH.exists():
        return {"chats": [], "active_id": None}
    try:
        payload = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"chats": [], "active_id": None}
    payload.setdefault("chats", [])
    return payload


def _write(payload: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = STORE_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(STORE_PATH)


def state() -> dict:
    with _LOCK:
        return _read()


def list_chats() -> list[dict]:
    return sorted(state()["chats"], key=lambda chat: chat.get("updated_at", ""), reverse=True)


def active_chat() -> dict | None:
    payload = state()
    active_id = payload.get("active_id")
    for chat in payload["chats"]:
        if chat["id"] == active_id:
            return chat
    return payload["chats"][-1] if payload["chats"] else None


def create_chat(title: str = "Новый чат") -> dict:
    with _LOCK:
        payload = _read()
        chat = {
            "id": uuid.uuid4().hex[:12],
            "title": title,
            "created_at": _now(),
            "updated_at": _now(),
            "messages": [],
        }
        payload["chats"].append(chat)
        payload["active_id"] = chat["id"]
        _write(payload)
        return chat


def set_active(chat_id: str) -> None:
    with _LOCK:
        payload = _read()
        payload["active_id"] = chat_id
        _write(payload)


def delete_chat(chat_id: str) -> None:
    with _LOCK:
        payload = _read()
        payload["chats"] = [chat for chat in payload["chats"] if chat["id"] != chat_id]
        if payload.get("active_id") == chat_id:
            payload["active_id"] = payload["chats"][-1]["id"] if payload["chats"] else None
        _write(payload)


def clear_all() -> None:
    with _LOCK:
        _write({"chats": [], "active_id": None})


def append_message(chat_id: str, message: dict) -> None:
    with _LOCK:
        payload = _read()
        for chat in payload["chats"]:
            if chat["id"] != chat_id:
                continue
            chat["messages"].append(message)
            chat["updated_at"] = _now()
            if chat["title"] == "Новый чат" and message.get("role") == "user":
                text = str(message.get("content", "")).strip().replace("\n", " ")
                chat["title"] = (text[:57] + "…") if len(text) > 60 else text or "Новый чат"
            break
        _write(payload)


def rename_chat(chat_id: str, title: str) -> None:
    with _LOCK:
        payload = _read()
        for chat in payload["chats"]:
            if chat["id"] == chat_id:
                chat["title"] = title.strip()[:80] or chat["title"]
                chat["updated_at"] = _now()
                break
        _write(payload)


def export_chat(chat: dict) -> str:
    lines = [f"# {chat['title']}", ""]
    for message in chat.get("messages", []):
        role = "Студент" if message.get("role") == "user" else "StudyMate"
        lines.append(f"**{role}:** {message.get('content', '')}")
        lines.append("")
        meta = message.get("meta") or {}
        if meta.get("sources"):
            lines.append("Источники:")
            for index, source in enumerate(meta["sources"], start=1):
                lines.append(
                    f"{index}. {source.get('title')} — стр. {source.get('page')}"
                )
            lines.append("")
    return "\n".join(lines)
