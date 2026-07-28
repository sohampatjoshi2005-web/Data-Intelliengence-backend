from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st

from .config import settings


def _ensure_db_parent(path: str) -> str:
    db_path = Path(path).expanduser()
    if not db_path.is_absolute():
        db_path = (Path.cwd() / db_path).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return str(db_path)


def create_sqlite_connection() -> sqlite3.Connection:
    resolved_path = _ensure_db_parent(settings.db_path)
    conn = sqlite3.connect(resolved_path, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


@st.cache_resource
def get_db_connection() -> sqlite3.Connection:
    return create_sqlite_connection()


def init_db(db_conn: sqlite3.Connection) -> None:
    db_conn.execute(
        """CREATE TABLE IF NOT EXISTS interaction_logs
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_email TEXT, channel TEXT, direction TEXT,
        content TEXT, classification TEXT, strategy TEXT,
        timestamp TEXT)"""
    )
    db_conn.execute(
        """CREATE TABLE IF NOT EXISTS customer_profiles
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, email TEXT UNIQUE, phone TEXT UNIQUE,
        address TEXT, gender TEXT, occupation TEXT,
        birth_date TEXT, social_handle TEXT,
        created_at TEXT)"""
    )
    db_conn.execute(
        """CREATE TABLE IF NOT EXISTS tickets
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_email TEXT, subject TEXT, priority TEXT,
        status TEXT, assigned_to TEXT, work_notes TEXT,
        due_date TEXT, created_at TEXT)"""
    )
    db_conn.commit()


def log_interaction(
    db_conn: sqlite3.Connection,
    email: str,
    channel: str,
    direction: str,
    content: str,
    classification: str = "N/A",
    strategy: str = "N/A",
) -> None:
    db_conn.execute(
        """INSERT INTO interaction_logs
        (customer_email, channel, direction, content, classification, strategy, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (email, channel, direction, content, classification, strategy, datetime.now().isoformat()),
    )
    db_conn.commit()
