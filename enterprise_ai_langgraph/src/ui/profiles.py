from __future__ import annotations

import pandas as pd
import streamlit as st


def render_customer_profiles(db_conn) -> None:
    st.title("👤 Unified CRM Profiles")
    profiles = pd.read_sql_query("SELECT * FROM customer_profiles", db_conn)
    st.dataframe(profiles, use_container_width=True)
