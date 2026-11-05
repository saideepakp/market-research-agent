"""Helpers shared by tools.py and derivatives.py."""

import json

from usage import LimitReached

MAX_CHARS = 6000  # cap on any single tool result
HTTP_TIMEOUT = 20


def _out(data) -> str:
    """Serialize a result and trim it so one tool call can't flood the context."""
    text = json.dumps(data, default=str, ensure_ascii=False)
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + '..."[truncated]"'


def _err(source: str, e: Exception) -> str:
    if isinstance(e, LimitReached):
        return _out({"error": str(e), "limit_reached": True})
    return _out({"error": f"{source} failed: {type(e).__name__}: {e}"})
