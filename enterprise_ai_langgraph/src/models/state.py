from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    input_text: str
    user_name: str
    user_email: str
    user_phone: str
    clean_text: str
    intent: str
    priority: str
    ticket_id: int
    due_date: str
    context_summary: str
    kb_context: str
    strategy: str
    drafts: str
    crm: dict[str, Any]
