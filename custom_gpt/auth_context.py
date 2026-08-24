from __future__ import annotations

import base64
import json
from typing import Any, Dict

import streamlit as st

from custom_gpt.config import settings
from custom_gpt.types import AuthContext

try:
    from jose import jwt
except Exception:
    jwt = None

_AUTH_KEY = "custom_gpt_auth_context"


def _decode_jwt_payload_noverify(token: str) -> Dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        payload_bytes = base64.urlsafe_b64decode(padded.encode("utf-8"))
        return json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return {}


def _decode_claims(token: str) -> Dict[str, Any]:
    if not token:
        return {}
    if jwt is not None and settings.jwt_secret:
        try:
            return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_alg])
        except Exception:
            pass
    return _decode_jwt_payload_noverify(token)


def get_auth_context() -> AuthContext:
    raw = st.session_state.get(_AUTH_KEY)
    if isinstance(raw, AuthContext):
        return raw
    if isinstance(raw, dict):
        ctx = AuthContext(**raw)
    else:
        ctx = AuthContext(tenant_id=settings.default_tenant, role=settings.default_role)
    st.session_state[_AUTH_KEY] = ctx.model_dump()
    return ctx


def set_auth_context(ctx: AuthContext) -> None:
    st.session_state[_AUTH_KEY] = ctx.model_dump()


def clear_auth_context() -> None:
    set_auth_context(AuthContext(tenant_id=settings.default_tenant, role=settings.default_role))


def apply_token(token: str) -> AuthContext:
    claims = _decode_claims(token)
    scopes = claims.get("scopes") or claims.get("scope") or []
    if isinstance(scopes, str):
        scopes = [s for s in scopes.split() if s]
    ctx = AuthContext(
        token=token,
        user_id=str(claims.get("sub") or claims.get("user_id") or ""),
        tenant_id=str(claims.get("tenant") or claims.get("tenant_id") or settings.default_tenant),
        role=str(claims.get("role") or settings.default_role),
        scopes=list(scopes),
        authenticated=bool(token),
        auth_source="jwt",
    )
    set_auth_context(ctx)
    return ctx
