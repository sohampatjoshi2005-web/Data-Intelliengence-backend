from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AuthContext(BaseModel):
    token: str = ""
    user_id: str = ""
    tenant_id: str = ""
    role: str = "viewer"
    scopes: List[str] = Field(default_factory=list)
    authenticated: bool = False
    auth_source: str = "none"


class HeaderConfig(BaseModel):
    apply_role_headers: bool = True
    include_api_key: bool = False
    api_key: Optional[str] = None
