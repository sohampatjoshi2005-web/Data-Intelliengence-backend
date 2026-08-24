from __future__ import annotations

from typing import Dict

from custom_gpt.auth_context import get_auth_context
from custom_gpt.types import HeaderConfig


def build_headers(header_cfg: HeaderConfig) -> Dict[str, str]:
    ctx = get_auth_context()
    headers: Dict[str, str] = {}

    if ctx.token:
        headers["Authorization"] = f"Bearer {ctx.token}"

    if header_cfg.apply_role_headers:
        if ctx.tenant_id:
            headers["X-Tenant-ID"] = ctx.tenant_id
        if ctx.role:
            headers["X-Role"] = ctx.role

    if header_cfg.include_api_key and header_cfg.api_key:
        headers["X-API-Key"] = header_cfg.api_key

    return headers
