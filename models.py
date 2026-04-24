"""
models.py — Thin helper layer over raw SQLite rows.

These are not ORMs — just functions that return dicts or sqlite3.Row objects
to keep things simple on a Pi. Each section corresponds to a DB table.
"""

from __future__ import annotations
import sqlite3
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

def get_role_by_name(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM roles WHERE name = ?", (name,)).fetchone()


def get_all_roles(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM roles ORDER BY id").fetchall()


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def get_user_by_id(conn: sqlite3.Connection, user_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT u.*, r.name AS role_name FROM users u "
        "JOIN roles r ON r.id = u.role_id WHERE u.id = ?",
        (user_id,),
    ).fetchone()


def get_user_by_username(conn: sqlite3.Connection, username: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT u.*, r.name AS role_name FROM users u "
        "JOIN roles r ON r.id = u.role_id WHERE u.username = ?",
        (username,),
    ).fetchone()


def get_user_by_email(conn: sqlite3.Connection, email: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT u.*, r.name AS role_name FROM users u "
        "JOIN roles r ON r.id = u.role_id WHERE u.email = ?",
        (email,),
    ).fetchone()


def get_all_users(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT u.*, r.name AS role_name FROM users u "
        "JOIN roles r ON r.id = u.role_id ORDER BY u.id"
    ).fetchall()


def create_user(
    conn: sqlite3.Connection,
    username: str,
    password_hash: str,
    role_id: int,
    display_name: str = "",
    email: str = "",
    created_by: int | None = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO users(username, display_name, email, password_hash,
                             role_id, created_by)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (username, display_name, email or None, password_hash, role_id, created_by),
    )
    conn.commit()
    return cur.lastrowid


def update_user(conn: sqlite3.Connection, user_id: int, **fields: Any) -> None:
    allowed = {"display_name", "email", "password_hash", "role_id", "is_active",
               "remember_token"}
    cols = {k: v for k, v in fields.items() if k in allowed}
    if not cols:
        return
    set_clause = ", ".join(f"{k} = ?" for k in cols)
    set_clause += ", updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')"
    conn.execute(
        f"UPDATE users SET {set_clause} WHERE id = ?",
        list(cols.values()) + [user_id],
    )
    conn.commit()


def delete_user(conn: sqlite3.Connection, user_id: int) -> None:
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Enrolled people
# ---------------------------------------------------------------------------

def get_all_enrolled(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT p.*, COUNT(i.id) AS image_count "
        "FROM enrolled_people p "
        "LEFT JOIN enrolled_images i ON i.person_id = p.id "
        "GROUP BY p.id ORDER BY p.name"
    ).fetchall()


def get_enrolled_by_id(conn: sqlite3.Connection, person_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM enrolled_people WHERE id = ?", (person_id,)
    ).fetchone()


def create_enrolled_person(
    conn: sqlite3.Connection,
    name: str,
    notes: str = "",
    created_by: int | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO enrolled_people(name, notes, created_by) VALUES (?, ?, ?)",
        (name, notes, created_by),
    )
    conn.commit()
    return cur.lastrowid


def update_enrolled_person(conn: sqlite3.Connection, person_id: int, **fields: Any) -> None:
    allowed = {"name", "notes", "is_active"}
    cols = {k: v for k, v in fields.items() if k in allowed}
    if not cols:
        return
    set_clause = ", ".join(f"{k} = ?" for k in cols)
    set_clause += ", updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')"
    conn.execute(
        f"UPDATE enrolled_people SET {set_clause} WHERE id = ?",
        list(cols.values()) + [person_id],
    )
    conn.commit()


def delete_enrolled_person(conn: sqlite3.Connection, person_id: int) -> None:
    conn.execute("DELETE FROM enrolled_people WHERE id = ?", (person_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Enrolled images
# ---------------------------------------------------------------------------

def get_images_for_person(conn: sqlite3.Connection, person_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM enrolled_images WHERE person_id = ? ORDER BY uploaded_at",
        (person_id,),
    ).fetchall()


def add_enrolled_image(
    conn: sqlite3.Connection,
    person_id: int,
    filename: str,
    uploaded_by: int | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO enrolled_images(person_id, filename, uploaded_by) VALUES (?, ?, ?)",
        (person_id, filename, uploaded_by),
    )
    conn.commit()
    return cur.lastrowid


def delete_enrolled_image(conn: sqlite3.Connection, image_id: int) -> None:
    conn.execute("DELETE FROM enrolled_images WHERE id = ?", (image_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Face embeddings
# ---------------------------------------------------------------------------

def get_all_embeddings(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT e.*, p.name AS person_name "
        "FROM face_embeddings e "
        "JOIN enrolled_people p ON p.id = e.person_id "
        "WHERE p.is_active = 1"
    ).fetchall()


def add_embedding(
    conn: sqlite3.Connection,
    person_id: int,
    image_id: int | None,
    embedding_bytes: bytes,
    model_name: str = "buffalo_l",
) -> int:
    cur = conn.execute(
        "INSERT INTO face_embeddings(person_id, image_id, embedding, model_name) "
        "VALUES (?, ?, ?, ?)",
        (person_id, image_id, embedding_bytes, model_name),
    )
    conn.commit()
    return cur.lastrowid


def delete_embeddings_for_person(conn: sqlite3.Connection, person_id: int) -> None:
    conn.execute("DELETE FROM face_embeddings WHERE person_id = ?", (person_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def create_event(conn: sqlite3.Connection, **fields: Any) -> int:
    allowed = {
        "event_type", "person_id", "person_name", "confidence",
        "snapshot_path", "occurred_at", "notes",
    }
    cols = {k: v for k, v in fields.items() if k in allowed and v is not None}
    col_names = ", ".join(cols)
    placeholders = ", ".join("?" for _ in cols)
    cur = conn.execute(
        f"INSERT INTO events({col_names}) VALUES ({placeholders})",
        list(cols.values()),
    )
    conn.commit()
    return cur.lastrowid


def update_event(conn: sqlite3.Connection, event_id: int, **fields: Any) -> None:
    allowed = {
        "recording_path", "recording_start", "recording_end",
        "webhook_sent", "webhook_error", "ended_at", "notes",
    }
    cols = {k: v for k, v in fields.items() if k in allowed}
    if not cols:
        return
    set_clause = ", ".join(f"{k} = ?" for k in cols)
    conn.execute(
        f"UPDATE events SET {set_clause} WHERE id = ?",
        list(cols.values()) + [event_id],
    )
    conn.commit()


def get_recent_events(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM events ORDER BY occurred_at DESC LIMIT ?", (limit,)
    ).fetchall()


def get_event_by_id(conn: sqlite3.Connection, event_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()


def count_events_today(conn: sqlite3.Connection, event_type: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM events "
        "WHERE event_type = ? AND date(occurred_at) = date('now')",
        (event_type,),
    ).fetchone()
    return row["n"] if row else 0


# ---------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------

def create_recording(conn: sqlite3.Connection, **fields: Any) -> int:
    allowed = {"event_id", "filename", "file_size_bytes", "duration_seconds",
               "started_at", "ended_at"}
    cols = {k: v for k, v in fields.items() if k in allowed and v is not None}
    col_names = ", ".join(cols)
    placeholders = ", ".join("?" for _ in cols)
    cur = conn.execute(
        f"INSERT INTO recordings({col_names}) VALUES ({placeholders})",
        list(cols.values()),
    )
    conn.commit()
    return cur.lastrowid


def get_all_recordings(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT r.*, e.event_type FROM recordings r "
        "LEFT JOIN events e ON e.id = r.event_id "
        "WHERE r.deleted = 0 ORDER BY r.started_at DESC"
    ).fetchall()


def soft_delete_recording(
    conn: sqlite3.Connection, recording_id: int, deleted_by: int | None = None
) -> None:
    conn.execute(
        "UPDATE recordings SET deleted=1, deleted_by=?, "
        "deleted_at=strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id=?",
        (deleted_by, recording_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Access schedules
# ---------------------------------------------------------------------------

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def get_schedules_for_person(conn: sqlite3.Connection, person_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM access_schedules WHERE person_id = ? AND is_active = 1 "
        "ORDER BY day_of_week, time_start",
        (person_id,),
    ).fetchall()


def create_schedule(
    conn: sqlite3.Connection,
    person_id: int,
    day_of_week: int,
    time_start: str,
    time_end: str,
) -> int:
    cur = conn.execute(
        "INSERT INTO access_schedules(person_id, day_of_week, time_start, time_end) "
        "VALUES (?, ?, ?, ?)",
        (person_id, day_of_week, time_start, time_end),
    )
    conn.commit()
    return cur.lastrowid


def delete_schedule(conn: sqlite3.Connection, schedule_id: int) -> None:
    conn.execute("DELETE FROM access_schedules WHERE id = ?", (schedule_id,))
    conn.commit()


def is_person_allowed_now(conn: sqlite3.Connection, person_id: int) -> bool:
    """
    Return True if the person has no schedules (unrestricted) or has a
    schedule entry that covers the current local day and time.
    """
    from datetime import datetime
    schedules = get_schedules_for_person(conn, person_id)
    if not schedules:
        return True   # no schedule = always allowed

    now = datetime.now()
    today = now.weekday()   # 0=Mon … 6=Sun
    current_time = now.strftime("%H:%M")

    for s in schedules:
        if s["day_of_week"] == today:
            if s["time_start"] <= current_time <= s["time_end"]:
                return True
    return False


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def add_audit(
    conn: sqlite3.Connection,
    action: str,
    user_id: int | None = None,
    username: str = "",
    target_type: str = "",
    target_id: str = "",
    detail: str = "",
    ip_address: str = "",
) -> None:
    conn.execute(
        """INSERT INTO audit_log
               (user_id, username, action, target_type, target_id, detail, ip_address)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (user_id, username, action, target_type, target_id, detail, ip_address),
    )
    conn.commit()


def get_audit_log(conn: sqlite3.Connection, limit: int = 200) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM audit_log ORDER BY occurred_at DESC LIMIT ?", (limit,)
    ).fetchall()
