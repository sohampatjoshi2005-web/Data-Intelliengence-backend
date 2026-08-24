from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class CustomGPTSettings:
    jwt_secret: str = os.getenv("CUSTOM_GPT_JWT_SECRET", "")
    jwt_alg: str = os.getenv("CUSTOM_GPT_JWT_ALG", "HS256")
    default_role: str = os.getenv("CUSTOM_GPT_DEFAULT_ROLE", "viewer")
    default_tenant: str = os.getenv("CUSTOM_GPT_DEFAULT_TENANT", "default")


settings = CustomGPTSettings()
