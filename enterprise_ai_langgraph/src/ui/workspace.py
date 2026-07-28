from __future__ import annotations

from datetime import datetime

import streamlit as st

from ..database import log_interaction
from ..graph.workflow import AgentRuntime, build_workflow
from ..services.communications import send_via_brevo
from ..services.tickets import create_action_ticket


def render_ai_workspace(db_conn, brevo_client, chroma_collection, analyzer, llm_client) -> None:
    st.title("📧 AI Agent Workflow")

    with st.sidebar:
        st.header("Knowledge Base")
        kb_input = st.text_area("Add Doc Segment")
        if st.button("Index Knowledge"):
            chroma_collection.add(documents=[kb_input], ids=[f"id_{datetime.now().timestamp()}"])
            st.success("Indexed!")

    c1, c2 = st.columns([1, 1])
    with c1:
        with st.container(border=True):
            st.subheader("Customer Details (CRM Ingestion)")
            col_a, col_b = st.columns(2)
            u_name = col_a.text_input("Name", value="Customer")
            u_email = col_b.text_input("Email")
            u_phone = col_a.text_input("Phone", value="555-0000")
            u_social = col_b.text_input("Social Media Handle")

            with st.expander("Additional CRM Data"):
                u_addr = st.text_input("Address")
                col_c, col_d = st.columns(2)
                u_gender = col_c.selectbox("Gender", ["Other", "Male", "Female", "Prefer not to say"])
                u_occ = col_d.text_input("Occupation")
                u_birth = st.date_input("Birth Date", value=datetime(1990, 1, 1))

            u_body = st.text_area("Inquiry Content", height=150)
            process = st.button(" Execute Agent ")

    if process:
        runtime = AgentRuntime(analyzer, llm_client, chroma_collection, db_conn)
        workflow = build_workflow(runtime)
        with st.status("Agents Processing...") as status:
            result = workflow.invoke(
                {
                    "input_text": u_body,
                    "user_name": u_name,
                    "user_email": u_email,
                    "user_phone": u_phone,
                    "crm": {
                        "phone": u_phone,
                        "address": u_addr,
                        "gender": u_gender,
                        "occ": u_occ,
                        "birth": str(u_birth),
                        "social": u_social,
                    },
                }
            )
            status.update(
                label=f"Analysis Complete - Ticket #{result.get('ticket_id', 'N/A')} Created",
                state="complete",
            )

        st.session_state["active_analysis"] = {
            "intent": result.get("intent", "N/A"),
            "ctx": result.get("context_summary", ""),
            "drafts": result.get("drafts", ""),
            "email": u_email,
            "name": u_name,
            "strat": result.get("strategy", "N/A"),
            "ticket_id": result.get("ticket_id"),
            "crm": result.get("crm", {}),
        }

    if "active_analysis" in st.session_state:
        with c2:
            data = st.session_state["active_analysis"]
            st.subheader("Intelligence Output")
            st.info(f"**Ticket ID:** #{data['ticket_id']} | **Intent & Urgency:** {data['intent']}")
            final_polish = st.text_area("Review/Edit Draft", data["drafts"], height=250)

            btn_col1, btn_col2, btn_col3 = st.columns(3)
            if btn_col1.button("✅ Send Email (Brevo)"):
                sent, error = send_via_brevo(
                    brevo_client,
                    data["email"],
                    f"Re: Inquiry [Ticket #{data['ticket_id']}]",
                    final_polish,
                )
                if sent:
                    log_interaction(
                        db_conn,
                        data["email"],
                        "Email",
                        "Outbound",
                        final_polish,
                        data["intent"],
                        data["strat"],
                    )
                    db_conn.execute(
                        """INSERT OR REPLACE INTO customer_profiles
                        (name, email, phone, address, gender, occupation, birth_date, social_handle, created_at)
                        VALUES (?,?,?,?,?,?,?,?,?)""",
                        (
                            data["name"],
                            data["email"],
                            data["crm"].get("phone", ""),
                            data["crm"].get("address", ""),
                            data["crm"].get("gender", "Other"),
                            data["crm"].get("occ", ""),
                            data["crm"].get("birth", ""),
                            data["crm"].get("social", ""),
                            datetime.now().isoformat(),
                        ),
                    )
                    db_conn.commit()
                    st.success("Email Sent & CRM Updated!")
                else:
                    st.error(f"Brevo API Error: {error}")

            if btn_col2.button(" View Open Ticket"):
                st.write(f"Ticket #{data['ticket_id']} is active in Manager.")

            if btn_col3.button("📅 Schedule Follow-up"):
                create_action_ticket(
                    db_conn,
                    data["email"],
                    f"7-Day Check for #{data['ticket_id']}",
                    "Low",
                    days_lead=7,
                )
                st.info("Follow-up Scheduled.")
