"""
auth.py — Authentication, session management, RBAC, and rate limiting.

Provides:
  - login / logout helpers
  - @login_required decorator
  - @role_required decorator
  - login attempt tracking (rate limiting / lockout)
  - password hashing via bcrypt
  - audit helpers
"""

from __future__ import annotations

import logging
import functools
from datetime import datetime, timezone, timedelta

from flask import (
    Blueprint, render_template, redirect, url_for, request,
    session, flash, g, current_app,
)
import bcrypt as _bcrypt

import config
import models
from database import get_db
from utils import get_client_ip, audit

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)

# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def hash_password(plaintext: str) -> str:
    return _bcrypt.hashpw(plaintext.encode(), _bcrypt.gensalt()).decode()


def verify_password(plaintext: str, hashed: str) -> bool:
    try:
        return _bcrypt.checkpw(plaintext.encode(), hashed.encode())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Rate limiting helpers (DB-backed, no Redis needed)
# ---------------------------------------------------------------------------

def _count_recent_failures(db, identifier: str) -> int:
    cutoff = (
        datetime.now(timezone.utc) - timedelta(seconds=config.LOGIN_LOCKOUT_SECONDS)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = db.execute(
        "SELECT COUNT(*) AS n FROM login_attempts "
        "WHERE identifier = ? AND success = 0 AND attempted_at >= ?",
        (identifier, cutoff),
    ).fetchone()
    return row["n"] if row else 0


def _record_attempt(db, identifier: str, success: bool) -> None:
    db.execute(
        "INSERT INTO login_attempts(identifier, success) VALUES (?, ?)",
        (identifier, 1 if success else 0),
    )
    db.commit()


def is_locked_out(db, identifier: str) -> bool:
    return _count_recent_failures(db, identifier) >= config.LOGIN_MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def login_user(user_row, remember: bool = False) -> None:
    """Populate the Flask session after a successful login."""
    session.clear()
    session["user_id"]   = user_row["id"]
    session["username"]  = user_row["username"]
    session["role"]      = user_row["role_name"]
    session["logged_in"] = True
    if remember:
        # Let the session outlive the browser close
        session.permanent = True


def logout_user() -> None:
    session.clear()


def current_user():
    """Return the current user row (cached on g) or None."""
    if hasattr(g, "_current_user"):
        return g._current_user
    user_id = session.get("user_id")
    if not user_id:
        g._current_user = None
        return None
    db = get_db()
    g._current_user = models.get_user_by_id(db, user_id)
    return g._current_user


def is_authenticated() -> bool:
    return bool(session.get("logged_in") and current_user())


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def login_required(f):
    """Redirect to login page if not authenticated."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not is_authenticated():
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("auth.login", next=request.path))
        return f(*args, **kwargs)
    return decorated


def role_required(*roles: str):
    """Allow only users whose role is in the given set."""
    def decorator(f):
        @functools.wraps(f)
        @login_required
        def decorated(*args, **kwargs):
            user_role = session.get("role", "")
            if user_role not in roles:
                flash("You do not have permission to access that page.", "danger")
                return redirect(url_for("admin.dashboard"))
            return f(*args, **kwargs)
        return decorated
    return decorator


# Convenience aliases
admin_required    = role_required("admin")
operator_required = role_required("admin", "operator")
viewer_required   = role_required("admin", "operator", "viewer")


# ---------------------------------------------------------------------------
# Login / logout routes
# ---------------------------------------------------------------------------

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if is_authenticated():
        return redirect(url_for("admin.dashboard"))

    if request.method == "POST":
        username  = request.form.get("username", "").strip()
        password  = request.form.get("password", "")
        remember  = bool(request.form.get("remember"))
        client_ip = get_client_ip()

        db = get_db()

        # Check lockout by both username and IP
        for ident in (username, client_ip):
            if is_locked_out(db, ident):
                flash(
                    "Too many failed attempts. Please wait before trying again.",
                    "danger",
                )
                audit(
                    "LOGIN_LOCKED_OUT",
                    detail=f"Identifier: {ident}",
                    ip_address=client_ip,
                )
                return render_template("auth/login.html"), 429

        user = models.get_user_by_username(db, username)

        if not user or not verify_password(password, user["password_hash"]):
            _record_attempt(db, username, False)
            _record_attempt(db, client_ip, False)
            audit(
                "LOGIN_FAILED",
                username=username,
                detail="Invalid credentials",
                ip_address=client_ip,
            )
            flash("Invalid username or password.", "danger")
            return render_template("auth/login.html"), 401

        if not user["is_active"]:
            audit(
                "LOGIN_DISABLED",
                user_id=user["id"],
                username=username,
                ip_address=client_ip,
            )
            flash("Your account has been disabled.", "danger")
            return render_template("auth/login.html"), 403

        _record_attempt(db, username, True)
        login_user(user, remember=remember)
        audit(
            "LOGIN_SUCCESS",
            user_id=user["id"],
            username=username,
            ip_address=client_ip,
        )
        logger.info("User %s logged in from %s", username, client_ip)

        next_url = request.args.get("next") or url_for("admin.dashboard")
        # Basic open-redirect protection — only allow relative paths
        if not next_url.startswith("/"):
            next_url = url_for("admin.dashboard")
        return redirect(next_url)

    return render_template("auth/login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    username  = session.get("username", "")
    user_id   = session.get("user_id")
    client_ip = get_client_ip()
    logout_user()
    audit("LOGOUT", user_id=user_id, username=username, ip_address=client_ip)
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))
