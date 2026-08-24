from __future__ import annotations

import pandas as pd
import streamlit as st


def render_shared_inbox(db_conn) -> None:
    st.title("📥 Shared Centralized Inbox")
    logs = pd.read_sql_query("SELECT * FROM interaction_logs ORDER BY timestamp DESC", db_conn)
    st.dataframe(logs, use_container_width=True)
