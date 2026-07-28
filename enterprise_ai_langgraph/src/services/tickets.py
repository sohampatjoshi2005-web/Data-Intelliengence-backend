from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from ..database import create_sqlite_connection


def create_action_ticket(
    db_conn: sqlite3.Connection | None,
    email: str,
    subject: str,
    priority: str = "Medium",
    days_lead: int = 3,
) -> tuple[int, str]:
    due_date = (datetime.now() + timedelta(days=days_lead)).strftime("%Y-%m-%d")
    connection = db_conn if db_conn is not None else create_sqlite_connection()
    own_connection = db_conn is None

    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO tickets (customer_email, subject, priority, status, assigned_to, work_notes, due_date, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                email,
                subject,
                priority,
                "OPEN",
                "Unassigned",
                "Ticket initialized by AI.",
                due_date,
                datetime.now().isoformat(),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid), due_date
    except sqlite3.OperationalError:
        # Retry with a fresh connection; LangGraph execution may run in a different worker context.
        retry_conn = create_sqlite_connection()
        retry_cursor = retry_conn.cursor()
        retry_cursor.execute(
            """
            INSERT INTO tickets (customer_email, subject, priority, status, assigned_to, work_notes, due_date, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                email,
                subject,
                priority,
                "OPEN",
                "Unassigned",
                "Ticket initialized by AI.",
                due_date,
                datetime.now().isoformat(),
            ),
        )
        retry_conn.commit()
        inserted_id = int(retry_cursor.lastrowid)
        retry_conn.close()
        return inserted_id, due_date
    finally:
        if own_connection:
            connection.close()
