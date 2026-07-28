from __future__ import annotations

import io
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from ..config import settings
from ..database import log_interaction

try:
    from streamlit_mic_recorder import mic_recorder
except Exception:  # pragma: no cover
    mic_recorder = None


def render_support_chatbot(db_conn, llm_client, whisper_model) -> None:
    st.title("🤖 Intelligent Assistant")

    p = ""
    if mic_recorder is None:
        st.info("Voice input is unavailable. Install `streamlit-mic-recorder` to enable it.")
    else:
        st.write("Click and hold to record, or use the text input below:")
        audio_data = mic_recorder(
            start_prompt="🎤 Start Recording",
            stop_prompt="🛑 Stop & Transcribe",
            key="voice_input",
        )

        if audio_data:
            audio_bytes = audio_data["bytes"]
            audio_file = io.BytesIO(audio_bytes)
            temp_file = f"temp_audio_{datetime.now().timestamp()}.wav"
            with open(temp_file, "wb") as file_handle:
                file_handle.write(audio_file.read())
            with st.spinner("Transcribing your voice..."):
                segments, _ = whisper_model.transcribe(temp_file, beam_size=5)
                p = " ".join([segment.text for segment in segments])
                st.success(f"Transcribed: {p}")
            try:
                os.remove(temp_file)
            except OSError:
                pass

    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = []

    for msg in st.session_state.chat_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    chat_input = st.chat_input("Query logs or tickets...")
    user_query = chat_input if chat_input else p

    if not user_query:
        return

    st.session_state.chat_msgs.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    t_details = pd.read_sql_query(
        "SELECT id, customer_email, status, assigned_to FROM tickets ORDER BY id DESC LIMIT 10",
        db_conn,
    )
    recent_logs = pd.read_sql_query(
        "SELECT customer_email, content, timestamp FROM interaction_logs ORDER BY timestamp DESC LIMIT 5",
        db_conn,
    )

    ticket_str = t_details.to_string(index=False)
    log_str = recent_logs.to_string(index=False)
    sys_ctx = (
        f"System Data:\nTICKETS:\n{ticket_str}\n\n"
        f"LOGS:\n{log_str}\n\n"
        f"User Question: {user_query}"
    )

    with st.chat_message("assistant"):
        stream = llm_client.chat.completions.create(
            model=settings.model_name,
            messages=[{"role": "user", "content": sys_ctx}],
            stream=True,
        )
        response_content = st.write_stream(stream)

    st.session_state.chat_msgs.append({"role": "assistant", "content": response_content})
    log_interaction(db_conn, "internal_bot", "Chatbot", "Internal", response_content)
