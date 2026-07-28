from __future__ import annotations

import os
from typing import Any, Dict

from app.core.config import settings

try:
    import ray
except Exception:
    ray = None


def get_orchestration_status() -> Dict[str, Any]:
    backend = settings.distributed_backend
    if backend == "auto":
        backend = "ray" if ray is not None else "local"

    out = {
        "enabled": bool(settings.distributed_enabled),
        "backend": backend,
        "ray_available": ray is not None,
        "mode": "local",
        "workers": int(os.getenv("DISTRIBUTED_WORKERS", "2")),
    }

    if not settings.distributed_enabled:
        return out

    if backend == "ray" and ray is not None:
        try:
            if not ray.is_initialized():
                ray.init(address=os.getenv("RAY_ADDRESS", "auto"), ignore_reinit_error=True, logging_level="ERROR")
            out["mode"] = "ray_cluster"
            out["resources"] = ray.available_resources()
        except Exception as exc:
            out["mode"] = "local_fallback"
            out["warning"] = str(exc)
    else:
        out["mode"] = "local_parallel"

    return out
