from __future__ import annotations

import pandas as pd
import streamlit as st


def render_analytics(db_conn) -> None:
    st.title("📊 Data Analytics & Insights")
    t_df = pd.read_sql_query("SELECT * FROM tickets", db_conn)
    if t_df.empty:
        st.info("Insufficient data for analytics.")
        return

    st.subheader("Ticket Volume by Priority")
    st.bar_chart(t_df["priority"].value_counts())
    st.subheader("Ticket Status Distribution")
    st.write(t_df["status"].value_counts())
