from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import smtplib
import ssl
import time
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Any, Literal


def _parse_bool_env(value: str | None) -> bool:
    raw = (value or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def auth_enabled() -> bool:
    return _parse_bool_env(os.environ.get("STOCK_SCREENER_AUTH_ENABLED"))


def auth_secret() -> bytes:
    secret = os.environ.get("STOCK_SCREENER_AUTH_SECRET", "").strip()
    if not secret:
        raise RuntimeError("missing STOCK_SCREENER_AUTH_SECRET")
    return secret.encode("utf-8")


def token_ttl_seconds() -> int:
    raw = os.environ.get("STOCK_SCREENER_AUTH_TOKEN_TTL_SECONDS", "").strip()
    if not raw:
        return 60 * 60 * 24 * 30
    try:
        ttl = int(raw)
    except ValueError as e:
        raise RuntimeError("invalid STOCK_SCREENER_AUTH_TOKEN_TTL_SECONDS") from e
    if ttl <= 0:
        raise RuntimeError("invalid STOCK_SCREENER_AUTH_TOKEN_TTL_SECONDS")
    return ttl


def normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_email(email: str) -> str:
    normalized = normalize_email(email)
    if not normalized or len(normalized) > 254 or _EMAIL_RE.fullmatch(normalized) is None:
        raise ValueError("invalid email")
    return normalized


AuthSignupMode = Literal["open", "email", "closed"]


def auth_signup_mode() -> AuthSignupMode:
    raw = os.environ.get("STOCK_SCREENER_AUTH_SIGNUP_MODE", "open").strip().lower()
    if raw in {"", "open"}:
        return "open"
    if raw in {"email", "email_verify", "email_verification"}:
        return "email"
    if raw in {"closed", "off", "disabled", "none"}:
        return "closed"
    raise RuntimeError("invalid STOCK_SCREENER_AUTH_SIGNUP_MODE")


def auth_allowed_email_domains() -> set[str] | None:
    raw = os.environ.get("STOCK_SCREENER_AUTH_ALLOWED_EMAIL_DOMAINS", "").strip()
    if not raw:
        return None
    domains = {d.strip().lower().lstrip("@") for d in raw.split(",") if d.strip()}
    return domains or None


def auth_email_code_ttl_seconds() -> int:
    raw = os.environ.get("STOCK_SCREENER_AUTH_EMAIL_CODE_TTL_SECONDS", "").strip()
    if not raw:
        return 10 * 60
    ttl = int(raw)
    if ttl <= 0:
        raise RuntimeError("invalid STOCK_SCREENER_AUTH_EMAIL_CODE_TTL_SECONDS")
    return ttl


def auth_email_code_cooldown_seconds() -> int:
    raw = os.environ.get("STOCK_SCREENER_AUTH_EMAIL_CODE_COOLDOWN_SECONDS", "").strip()
    if not raw:
        return 60
    cooldown = int(raw)
    if cooldown < 0:
        raise RuntimeError("invalid STOCK_SCREENER_AUTH_EMAIL_CODE_COOLDOWN_SECONDS")
    return cooldown


def auth_email_code_max_attempts() -> int:
    raw = os.environ.get("STOCK_SCREENER_AUTH_EMAIL_CODE_MAX_ATTEMPTS", "").strip()
    if not raw:
        return 5
    attempts = int(raw)
    if attempts <= 0:
        raise RuntimeError("invalid STOCK_SCREENER_AUTH_EMAIL_CODE_MAX_ATTEMPTS")
    return attempts


def auth_email_debug_return_code() -> bool:
    return _parse_bool_env(os.environ.get("STOCK_SCREENER_AUTH_EMAIL_DEBUG_RETURN_CODE"))


@dataclass(frozen=True)
class SmtpConfig:
    host: str
    port: int
    username: str | None
    password: str | None
    from_addr: str
    tls: bool
    ssl: bool


def smtp_config() -> SmtpConfig | None:
    host = os.environ.get("STOCK_SCREENER_SMTP_HOST", "").strip()
    if not host:
        return None
    port_raw = os.environ.get("STOCK_SCREENER_SMTP_PORT", "").strip()
    port = int(port_raw) if port_raw else 587
    if port <= 0:
        raise RuntimeError("invalid STOCK_SCREENER_SMTP_PORT")

    username = os.environ.get("STOCK_SCREENER_SMTP_USERNAME", "").strip() or None
    password = os.environ.get("STOCK_SCREENER_SMTP_PASSWORD", "").strip() or None
    from_addr = os.environ.get("STOCK_SCREENER_SMTP_FROM", "").strip() or (username or "")
    if not from_addr:
        raise RuntimeError("missing STOCK_SCREENER_SMTP_FROM")

    tls = _parse_bool_env(os.environ.get("STOCK_SCREENER_SMTP_TLS", "1"))
    ssl_enabled = _parse_bool_env(os.environ.get("STOCK_SCREENER_SMTP_SSL"))
    if ssl_enabled:
        tls = False
    return SmtpConfig(
        host=host,
        port=port,
        username=username,
        password=password,
        from_addr=from_addr,
        tls=tls,
        ssl=ssl_enabled,
    )


def send_email(*, config: SmtpConfig, to_addr: str, subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["From"] = config.from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    client: smtplib.SMTP
    if config.ssl:
        client = smtplib.SMTP_SSL(config.host, config.port, timeout=10)
    else:
        client = smtplib.SMTP(config.host, config.port, timeout=10)

    try:
        client.ehlo()
        if config.tls and not config.ssl:
            ctx = ssl.create_default_context()
            client.starttls(context=ctx)
            client.ehlo()
        if config.username and config.password:
            client.login(config.username, config.password)
        client.send_message(msg)
    finally:
        try:
            client.quit()
        except Exception:
            try:
                client.close()
            except Exception:
                pass

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    raw = (data or "").strip()
    if not raw:
        raise ValueError("empty base64url")
    padded = raw + "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _sign(payload: bytes, secret: bytes) -> bytes:
    return hmac.new(secret, payload, hashlib.sha256).digest()


def create_access_token(claims: dict[str, Any], *, secret: bytes, expires_at: int) -> str:
    body = dict(claims)
    body["v"] = 1
    body["exp"] = int(expires_at)
    payload = json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = _sign(payload, secret)
    return f"{_b64url_encode(payload)}.{_b64url_encode(sig)}"


def decode_access_token(token: str, *, secret: bytes) -> dict[str, Any]:
    raw = (token or "").strip()
    if not raw:
        raise ValueError("empty token")
    parts = raw.split(".")
    if len(parts) != 2:
        raise ValueError("invalid token format")
    payload_b64, sig_b64 = parts
    payload = _b64url_decode(payload_b64)
    sig = _b64url_decode(sig_b64)
    expected = _sign(payload, secret)
    if not hmac.compare_digest(sig, expected):
        raise ValueError("invalid token signature")
    try:
        claims = json.loads(payload.decode("utf-8"))
    except Exception as e:
        raise ValueError("invalid token payload") from e
    if not isinstance(claims, dict):
        raise ValueError("invalid token payload")
    if int(claims.get("v") or 0) != 1:
        raise ValueError("unsupported token version")
    exp = int(claims.get("exp") or 0)
    if exp <= 0:
        raise ValueError("missing token exp")
    if int(time.time()) >= exp:
        raise ValueError("token expired")
    return claims


def hash_password(password: str) -> tuple[str, str]:
    pw = password or ""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, 200_000)
    return _b64url_encode(dk), _b64url_encode(salt)


def verify_password(password: str, password_hash: str, password_salt: str) -> bool:
    pw = password or ""
    salt = _b64url_decode(password_salt)
    expected = _b64url_decode(password_hash)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, 200_000)
    return hmac.compare_digest(dk, expected)


@dataclass(frozen=True)
class AuthUser:
    id: str
    username: str
    role: Literal["admin", "user"]
