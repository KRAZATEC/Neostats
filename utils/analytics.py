"""
NeoStats AI Chatbot — Analytics & Session Utilities
Tracks usage metrics, conversation statistics, and session state helpers.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SessionAnalytics:
    """
    Tracks per-session metrics: messages sent, tokens, latency, RAG hits, web searches.
    All data is ephemeral (session-scoped) — nothing is persisted to disk.
    """

    def __init__(self):
        self.session_start: float = time.time()
        self.total_messages: int = 0
        self.total_rag_hits: int = 0
        self.total_web_searches: int = 0
        self.total_response_time: float = 0.0
        self.provider_counts: dict = {}
        self.domain_used: str = "General"
        self.events: list[dict] = []

    def log_message(
        self,
        query: str,
        response_time: float,
        provider: str,
        rag_used: bool = False,
        web_used: bool = False,
        response_mode: str = "Detailed",
    ):
        """Record a completed chat exchange."""
        self.total_messages += 1
        self.total_response_time += response_time
        if rag_used:
            self.total_rag_hits += 1
        if web_used:
            self.total_web_searches += 1
        self.provider_counts[provider] = self.provider_counts.get(provider, 0) + 1

        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "query_length": len(query),
            "response_time_s": round(response_time, 2),
            "provider": provider,
            "rag_used": rag_used,
            "web_used": web_used,
            "response_mode": response_mode,
        })

    @property
    def avg_response_time(self) -> float:
        if self.total_messages == 0:
            return 0.0
        return round(self.total_response_time / self.total_messages, 2)

    @property
    def session_duration_mins(self) -> float:
        return round((time.time() - self.session_start) / 60, 1)

    def summary(self) -> dict:
        """Return a summary dict for display."""
        return {
            "total_messages": self.total_messages,
            "avg_response_time_s": self.avg_response_time,
            "rag_hits": self.total_rag_hits,
            "web_searches": self.total_web_searches,
            "session_duration_mins": self.session_duration_mins,
            "providers_used": self.provider_counts,
        }


def export_chat_history(history: list[dict], format: str = "txt") -> str:
    """
    Export chat history as a formatted string.

    Args:
        history: List of {role, content} message dicts
        format: 'txt' or 'json'

    Returns:
        Formatted export string
    """
    if format == "json":
        return json.dumps(history, indent=2)

    lines = [f"NeoStats AI Chat Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n", "="*60]
    for msg in history:
        role = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"\n[{role}]\n{msg['content']}")
    return "\n".join(lines)


def get_conversation_title(history: list[dict]) -> str:
    """Generate a short title from the first user message."""
    for msg in history:
        if msg["role"] == "user":
            text = msg["content"][:50].strip()
            return text + ("..." if len(msg["content"]) > 50 else "")
    return "New Conversation"
