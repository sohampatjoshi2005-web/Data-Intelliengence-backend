from __future__ import annotations

import streamlit as st

from src.clients import (
    get_analyzer,
    get_brevo_client,
    get_chroma_collection,
    get_openai_client,
    get_whisper_model,
)
from src.database import get_db_connection, init_db
from src.ui.analytics import render_analytics
from src.ui.chatbot import render_support_chatbot
from src.ui.inbox import render_shared_inbox
from src.ui.profiles import render_customer_profiles
from src.ui.tickets import render_ticket_manager
from src.ui.workspace import render_ai_workspace


def main() -> None:
    st.set_page_config(layout="wide", page_title="Enterprise AI Command Center", page_icon="🚀")

    db_conn = get_db_connection()
    init_db(db_conn)

    brevo_client = get_brevo_client()
    llm_client = get_openai_client()
    analyzer = get_analyzer()
    chroma_collection = get_chroma_collection()
    whisper_model = get_whisper_model()

    st.sidebar.title("🚀 Command Center")
    menu = st.sidebar.radio(
        "Navigation",
        [
            "AI Workspace",
            "Ticket Manager",
            "Shared Inbox",
            "Customer Profiles",
            "AutoML & Analytics",
            "Support Chatbot",
        ],
    )

    if menu == "AI Workspace":
        render_ai_workspace(db_conn, brevo_client, chroma_collection, analyzer, llm_client)
    elif menu == "Ticket Manager":
        render_ticket_manager(db_conn)
    elif menu == "AutoML & Analytics":
        render_analytics(db_conn)
    elif menu == "Shared Inbox":
        render_shared_inbox(db_conn)
    elif menu == "Support Chatbot":
        render_support_chatbot(db_conn, llm_client, whisper_model)
    elif menu == "Customer Profiles":
        render_customer_profiles(db_conn)


if __name__ == "__main__":
    main()
