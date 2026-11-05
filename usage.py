"""
Usage limits for the external data providers.

Every outgoing API request is counted in usage.db (a small SQLite file next to
this one). Before a request is made, we check it against per-minute, per-day
and per-month caps. If a cap is reached, the tool returns an error message
instead of calling the API, and the agent switches to another source.

The defaults sit safely BELOW each provider's free-plan limits, so you never
hit the provider's own limit (which could get your key throttled or blocked).

Override any limit in .env, e.g.  LIMIT_COINGECKO_MONTH=5000
Set a limit to 0 to block that provider entirely.
"""

import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

DB_PATH = os.getenv("USAGE_DB") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "usage.db")

# provider -> {window: max requests}.  None = no cap for that window.
# Every provider has a monthly cap. Where the provider publishes a monthly limit,
# ours sits below it; where it doesn't, the monthly cap is a safety budget.
#   Provider free-plan limits these protect (as of Nov 2026):
#   coingecko : 10,000 credits/month, 100/min (Demo plan); keyless API is stricter
#   finnhub   : 60/min
#   fred      : 120/min
#   sec       : 10/second fair-access rule
#   cftc      : public government API; throttles heavy keyless use
#   yfinance  : no published limit; Yahoo throttles heavy use
#   ddgs      : no published limit; search engines throttle heavy use
DEFAULT_LIMITS = {
    "coingecko": {"minute": 30,  "day": 400,  "month": 9000},
    "finnhub":   {"minute": 50,  "day": 2000, "month": 30000},
    "fred":      {"minute": 100, "day": 2000, "month": 20000},
    "sec":       {"minute": 120, "day": 1000, "month": 10000},
    "cftc":      {"minute": 30,  "day": 300,  "month": 3000},
    "yfinance":  {"minute": 60,  "day": 2000, "month": 30000},
    "ddgs":      {"minute": 10,  "day": 300,  "month": 6000},
}

# Hints the agent sees when a provider is capped, so it can switch source.
FALLBACK_HINTS = {
    "coingecko": "For crypto prices use get_quote / get_price_history with symbols like BTC-USD, ETH-USD.",
    "finnhub": "Use web_search with recent_days for company news instead.",
    "fred": "Use web_search for the latest macro figures instead.",
    "sec": "Use web_search to find the filing instead.",
    "cftc": "Use web_search for the latest Commitments of Traders figures instead.",
    "yfinance": "Use web_search for prices and figures, and say the data may be less precise.",
    "ddgs": "Answer with the structured data tools, and say web search is temporarily unavailable.",
}

_lock = threading.Lock()


class LimitReached(Exception):
    pass


def _limits(provider: str) -> dict:
    limits = dict(DEFAULT_LIMITS.get(provider, {}))
    # Keyless CoinGecko has a much lower rate limit than the Demo key.
    if provider == "coingecko" and not os.getenv("COINGECKO_API_KEY"):
        limits["minute"] = 5
    for window in ("minute", "day", "month"):
        override = os.getenv(f"LIMIT_{provider.upper()}_{window.upper()}")
        if override not in (None, ""):
            limits[window] = int(override)
    return limits


def _window_starts(now: float) -> dict:
    dt = datetime.fromtimestamp(now, timezone.utc)
    return {
        "minute": now - 60,                                   # rolling 60 seconds
        "day": dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp(),
        "month": dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp(),
    }


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("CREATE TABLE IF NOT EXISTS usage (provider TEXT, ts REAL)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage ON usage (provider, ts)")
    return conn


def _reset_text(window: str, now: float) -> str:
    dt = datetime.fromtimestamp(now, timezone.utc)
    if window == "minute":
        return "in under a minute"
    if window == "day":
        return "at midnight UTC"
    nxt = dt.replace(year=dt.year + (dt.month == 12), month=dt.month % 12 + 1, day=1)
    return f"on {nxt.date().isoformat()}"


def check_and_record(provider: str, n: int = 1) -> None:
    """Reserve n requests for a provider, or raise LimitReached.
    Check and record happen in one transaction, so parallel tool calls
    can't sneak past a limit together."""
    limits = _limits(provider)
    now = time.time()
    starts = _window_starts(now)
    with _lock:
        conn = _connect()
        try:
            conn.execute("BEGIN IMMEDIATE")  # lock across processes (app + MCP server)
            for window, cap in limits.items():
                if cap is None:
                    continue
                used = conn.execute("SELECT COUNT(*) FROM usage WHERE provider=? AND ts>=?",
                                    (provider, starts[window])).fetchone()[0]
                if used + n > cap:
                    conn.rollback()
                    label = {"minute": "per-minute", "day": "daily", "month": "monthly"}[window]
                    raise LimitReached(
                        f"{provider} {label} limit reached ({used}/{cap} requests). "
                        f"Resets {_reset_text(window, now)}. {FALLBACK_HINTS.get(provider, '')}"
                    )
            conn.executemany("INSERT INTO usage VALUES (?, ?)", [(provider, now)] * n)
            # keep the file small: drop records older than ~40 days
            conn.execute("DELETE FROM usage WHERE ts < ?", (now - 40 * 86400,))
            conn.commit()
        finally:
            conn.close()


def usage_summary() -> list[dict]:
    """Current usage per provider, for display in the UI."""
    now = time.time()
    starts = _window_starts(now)
    rows = []
    conn = _connect()
    try:
        for provider in DEFAULT_LIMITS:
            limits = _limits(provider)
            row = {"provider": provider}
            for window in ("day", "month"):
                used = conn.execute("SELECT COUNT(*) FROM usage WHERE provider=? AND ts>=?",
                                    (provider, starts[window])).fetchone()[0]
                row[window] = used
                row[f"{window}_limit"] = limits.get(window)
            rows.append(row)
    finally:
        conn.close()
    return rows
