# observability.py — Langfuse client singleton for ReqPal
# Silently disabled when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set.
#
# Usage:
#   from backend.services.observability import get_langfuse
#   lf = get_langfuse()          # None if keys missing
#   if lf:
#       trace = lf.trace(name="rag_answer", input={...})

from __future__ import annotations

import os
from typing import Optional, Any

_client: Optional[Any] = None
_initialized = False


def get_langfuse() -> Optional[Any]:
    """
    Return the Langfuse singleton, or None if keys are not configured.
    Safe to call on every request — initialises only once.
    """
    global _client, _initialized

    if _initialized:
        return _client

    _initialized = True

    pk   = (os.getenv("LANGFUSE_PUBLIC_KEY")  or "").strip()
    sk   = (os.getenv("LANGFUSE_SECRET_KEY")  or "").strip()
    host = (os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com").strip()

    if not pk or not sk:
        print("[Langfuse] Keys not set — observability disabled")
        return None

    try:
        from langfuse import Langfuse
        _client = Langfuse(public_key=pk, secret_key=sk, host=host)
        print(f"[Langfuse] Connected → {host}")
    except Exception as e:
        print(f"[Langfuse] Init failed: {e}")
        _client = None

    return _client
