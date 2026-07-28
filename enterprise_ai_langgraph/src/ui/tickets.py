from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st


def render_ticket_manager(db_conn) -> None:
    st.title("🎟️ Enhanced Ticket Queue")
    f_col1, _ = st.columns(2)
    status_filter = f_col1.multiselect(
        "Filter Status",
        ["OPEN", "IN PROGRESS", "PENDING WITH CUSTOMER", "CLOSED"],
        default=["OPEN", "IN PROGRESS", "PENDING WITH CUSTOMER"],
    )

    query = "SELECT * FROM tickets WHERE status IN ({}) ORDER BY due_date ASC".format(
        ",".join(["?"] * len(status_filter))
    )
    t_data = pd.read_sql_query(query, db_conn, params=status_filter)

    if t_data.empty:
        st.success("No tickets found for selected filters.")
        return

    options = ["OPEN", "IN PROGRESS", "PENDING WITH CUSTOMER", "CLOSED"]
    for _, row in t_data.iterrows():
        with st.expander(f"Ticket #{row['id']} | {row['status']} | Due: {row['due_date']}"):
            st.write(f"**Subject:** {row['subject']}")
            st.write(f"**Customer:** {row['customer_email']}")
            c_up1, c_up2, _ = st.columns(3)
            new_status = c_up1.selectbox(
                "Update Status",
                options,
                index=options.index(row["status"]),
                key=f"stat_{row['id']}",
            )
            new_assignee = c_up2.text_input(
                "Assigned Helpdesk Person",
                value=row["assigned_to"],
                key=f"assign_{row['id']}",
            )
            st.write("**Work Notes (Log of work done):**")
            st.caption(row["work_notes"])
            add_note = st.text_input("Add to Work Notes", key=f"note_{row['id']}")

            if st.button(f"Save Changes for #{row['id']}"):
                updated_notes = row["work_notes"]
                if add_note:
                    updated_notes = (
                        row["work_notes"]
                        + f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {add_note}"
                    )
                db_conn.execute(
                    "UPDATE tickets SET status=?, assigned_to=?, work_notes=? WHERE id=?",
                    (new_status, new_assignee, updated_notes, row["id"]),
                )
                db_conn.commit()
                st.rerun()
