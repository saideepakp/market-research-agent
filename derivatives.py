"""
Derivatives and trading-instrument tools: options, futures, FX, volatility,
and futures positioning (CFTC Commitments of Traders).

Sources (all permanently free, no key):
  - yfinance : option chains, futures, FX and volatility indices (Yahoo Finance;
               options data is delayed ~15 minutes)
  - CFTC     : Commitments of Traders reports, from the US Commodity Futures
               Trading Commission's public API (government data)
  - Greeks, expected moves and max pain are calculated locally (no API).

Every request is counted and capped by usage.py.
"""

import math
import os
import time
from datetime import date, datetime, timezone

import requests
import yfinance as yf
from langchain_core.tools import tool

from common import HTTP_TIMEOUT, _err, _out
from usage import check_and_record

# ==========================================================================
# Black-Scholes math (local, no API)
# ==========================================================================

def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _npdf(x: float) -> float:
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def black_scholes(spot, strike, t_years, iv, rate, kind="call", div_yield=0.0) -> dict | None:
    """Theoretical price and Greeks for a European option.
    iv, rate and div_yield are decimals (0.25 = 25%). Returns None if inputs are unusable.
    theta is per calendar day; vega and rho are per 1 percentage point."""
    if not all(v and v > 0 for v in (spot, strike, t_years, iv)):
        return None
    sq = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate - div_yield + iv * iv / 2) * t_years) / (iv * sq)
    d2 = d1 - iv * sq
    dq, dr = math.exp(-div_yield * t_years), math.exp(-rate * t_years)
    common_theta = -spot * dq * _npdf(d1) * iv / (2 * sq)
    if kind == "call":
        price = spot * dq * _ncdf(d1) - strike * dr * _ncdf(d2)
        delta = dq * _ncdf(d1)
        theta = common_theta - rate * strike * dr * _ncdf(d2) + div_yield * spot * dq * _ncdf(d1)
        rho = strike * t_years * dr * _ncdf(d2) / 100
        prob_itm = _ncdf(d2)
    else:
        price = strike * dr * _ncdf(-d2) - spot * dq * _ncdf(-d1)
        delta = dq * (_ncdf(d1) - 1)
        theta = common_theta + rate * strike * dr * _ncdf(-d2) - div_yield * spot * dq * _ncdf(-d1)
        rho = -strike * t_years * dr * _ncdf(-d2) / 100
        prob_itm = _ncdf(-d2)
    return {
        "price": price, "delta": delta,
        "gamma": dq * _npdf(d1) / (spot * iv * sq),
        "theta_per_day": theta / 365,
        "vega_per_1pt": spot * dq * _npdf(d1) * sq / 100,
        "rho_per_1pt": rho,
        "prob_itm": prob_itm,
    }


def _years_to_expiry(expiration: str) -> float:
    """US equity options stop trading at 4pm New York time (about 20:30 UTC on average)."""
    exp = datetime.strptime(expiration, "%Y-%m-%d").replace(hour=20, minute=30, tzinfo=timezone.utc)
    seconds = (exp - datetime.now(timezone.utc)).total_seconds()
    return max(seconds, 3600) / (365 * 86400)  # at least 1 hour, to keep the math stable


_rate_cache = {"value": None, "at": 0.0}


def _risk_free_rate() -> float:
    """13-week US Treasury bill yield (^IRX) as a decimal, cached for 6 hours.
    Falls back to RISK_FREE_RATE in .env (default 0.04) if it can't be fetched."""
    if _rate_cache["value"] is not None and time.time() - _rate_cache["at"] < 6 * 3600:
        return _rate_cache["value"]
    rate = float(os.getenv("RISK_FREE_RATE", "0.04"))
    try:
        check_and_record("yfinance")
        hist = yf.Ticker("^IRX").history(period="5d")
        if not hist.empty:
            rate = float(hist["Close"].iloc[-1]) / 100
    except Exception:
        pass
    _rate_cache.update(value=rate, at=time.time())
    return rate


# ==========================================================================
# Options chain analysis
# ==========================================================================

def _num(v):
    """NaN/None -> None, otherwise float."""
    try:
        v = float(v)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def _mid(row) -> float | None:
    bid, ask, last = _num(row.get("bid")), _num(row.get("ask")), _num(row.get("lastPrice"))
    if bid and ask and ask >= bid:
        return (bid + ask) / 2
    return last


def _max_pain(calls, puts) -> float | None:
    """Expiry price at which option holders' total payout is smallest."""
    c = [(_num(r["strike"]), _num(r.get("openInterest")) or 0) for _, r in calls.iterrows()]
    p = [(_num(r["strike"]), _num(r.get("openInterest")) or 0) for _, r in puts.iterrows()]
    strikes = sorted({k for k, _ in c + p if k is not None})
    if not strikes or not any(oi for _, oi in c + p):
        return None
    def payout(price):
        return (sum(oi * max(0.0, price - k) for k, oi in c) +
                sum(oi * max(0.0, k - price) for k, oi in p))
    return min(strikes, key=payout)


def _analyze_chain(calls, puts, spot, expiration, rate, strikes_around=None):
    """Summary stats for one expiration, plus near-the-money rows with Greeks."""
    t = _years_to_expiry(expiration)
    sum_col = lambda df, col: float(df[col].fillna(0).sum()) if col in df else 0.0
    call_vol, put_vol = sum_col(calls, "volume"), sum_col(puts, "volume")
    call_oi, put_oi = sum_col(calls, "openInterest"), sum_col(puts, "openInterest")

    strikes = sorted(set(calls["strike"]).intersection(puts["strike"])) or sorted(calls["strike"])
    atm = min(strikes, key=lambda k: abs(k - spot))
    atm_call = calls[calls["strike"] == atm].iloc[0] if (calls["strike"] == atm).any() else None
    atm_put = puts[puts["strike"] == atm].iloc[0] if (puts["strike"] == atm).any() else None
    ivs = [_num(r["impliedVolatility"]) for r in (atm_call, atm_put) if r is not None]
    ivs = [v for v in ivs if v and v > 0.01]  # Yahoo reports ~0 IV for dead quotes
    atm_iv = sum(ivs) / len(ivs) if ivs else None

    straddle = None
    if atm_call is not None and atm_put is not None:
        cm, pm = _mid(atm_call), _mid(atm_put)
        straddle = cm + pm if cm and pm else None

    def greeks_rows(df, kind):
        rows = []
        for _, r in df.iterrows():
            iv = _num(r.get("impliedVolatility"))
            g = black_scholes(spot, _num(r["strike"]), t, iv, rate, kind) if iv and iv > 0.01 else None
            rows.append({
                "strike": _num(r["strike"]), "last": _num(r.get("lastPrice")),
                "bid": _num(r.get("bid")), "ask": _num(r.get("ask")),
                "volume": int(_num(r.get("volume")) or 0),
                "open_interest": int(_num(r.get("openInterest")) or 0),
                "iv_pct": round(iv * 100, 1) if iv else None,
                "delta": round(g["delta"], 3) if g else None,
                "gamma": round(g["gamma"], 4) if g else None,
                "theta_per_day": round(g["theta_per_day"], 3) if g else None,
                "vega_per_1pt": round(g["vega_per_1pt"], 3) if g else None,
            })
        return rows

    # 25-delta skew: IV of the put nearest -0.25 delta minus the call nearest +0.25 delta
    skew = None
    if atm_iv:
        c_rows, p_rows = greeks_rows(calls, "call"), greeks_rows(puts, "put")
        c25 = [r for r in c_rows if r["delta"] is not None]
        p25 = [r for r in p_rows if r["delta"] is not None]
        if c25 and p25:
            c = min(c25, key=lambda r: abs(r["delta"] - 0.25))
            p = min(p25, key=lambda r: abs(r["delta"] + 0.25))
            skew = {"put_25d_iv_pct": p["iv_pct"], "call_25d_iv_pct": c["iv_pct"],
                    "skew_pts": round(p["iv_pct"] - c["iv_pct"], 1)}

    top = lambda df: _num(df.loc[df["openInterest"].fillna(0).idxmax(), "strike"]) \
        if "openInterest" in df and df["openInterest"].fillna(0).sum() > 0 else None

    summary = {
        "expiration": expiration,
        "days_to_expiry": round(t * 365, 1),
        "atm_strike": atm,
        "atm_iv_pct": round(atm_iv * 100, 1) if atm_iv else None,
        "expected_move": round(straddle, 2) if straddle else None,
        "expected_move_pct": round(straddle / spot * 100, 2) if straddle else None,
        "expected_range": [round(spot - straddle, 2), round(spot + straddle, 2)] if straddle else None,
        "put_call_volume_ratio": round(put_vol / call_vol, 2) if call_vol else None,
        "put_call_oi_ratio": round(put_oi / call_oi, 2) if call_oi else None,
        "call_volume": int(call_vol), "put_volume": int(put_vol),
        "call_open_interest": int(call_oi), "put_open_interest": int(put_oi),
        "max_pain": _max_pain(calls, puts),
        "highest_oi_call_strike": top(calls), "highest_oi_put_strike": top(puts),
        "skew_25_delta": skew,
    }
    if strikes_around is None:
        return summary, None

    near = sorted(sorted(strikes, key=lambda k: abs(k - spot))[: 2 * strikes_around + 1])
    rows = {
        "calls": greeks_rows(calls[calls["strike"].isin(near)].sort_values("strike"), "call"),
        "puts": greeks_rows(puts[puts["strike"].isin(near)].sort_values("strike"), "put"),
    }
    return summary, rows


def _spot_from_chain(chain, tk) -> float | None:
    u = chain.underlying or {}
    for k in ("regularMarketPrice", "postMarketPrice", "bid"):
        if _num(u.get(k)):
            return _num(u[k])
    try:
        return _num(tk.fast_info["last_price"])
    except Exception:
        return None


def _pick_expiration(expirations, wanted: str | None) -> str:
    """The available expiration closest to `wanted`; by default the first one at least 7 days out."""
    if wanted:
        target = datetime.strptime(wanted, "%Y-%m-%d").date()
        return min(expirations, key=lambda e: abs((date.fromisoformat(e) - target).days))
    later = [e for e in expirations if (date.fromisoformat(e) - date.today()).days >= 7]
    return later[0] if later else expirations[0]


OPTIONS_NOTE = ("Yahoo Finance options data, delayed ~15 min; bid/ask can be 0 outside market hours. "
                "Greeks use Black-Scholes with Yahoo's implied volatility and the 13-week T-bill rate; "
                "they are estimates. Expected move = at-the-money straddle price.")


@tool
def get_options_chain(ticker: str, expiration: str | None = None, strikes_around: int = 6) -> str:
    """Options chain for a stock, ETF or index (e.g. AAPL, SPY, QQQ, ^SPX) for ONE expiration.
    expiration: 'YYYY-MM-DD'; the closest listed date is used. Default: first expiration
    at least 7 days out. strikes_around: number of strikes above and below the current price.
    Returns, for calls and puts near the money: bid, ask, last, volume, open interest,
    implied volatility and Greeks (delta, gamma, theta, vega). Also a summary: at-the-money IV,
    expected move, put/call volume and open-interest ratios, max pain, highest-OI strikes,
    25-delta skew, and the list of available expirations."""
    try:
        check_and_record("yfinance", n=2)  # expiration list + chain
        tk = yf.Ticker(ticker)
        expirations = list(tk.options or [])
        if not expirations:
            return _out({"error": f"No listed options found for '{ticker}'."})
        exp = _pick_expiration(expirations, expiration)
        chain = tk.option_chain(exp)
        if chain.calls is None or chain.calls.empty:
            return _out({"error": f"No option quotes for {ticker} expiring {exp}."})
        spot = _spot_from_chain(chain, tk)
        if not spot:
            return _out({"error": f"Couldn't get the current price of {ticker}."})
        rate = _risk_free_rate()
        n = max(2, min(int(strikes_around), 12))
        summary, rows = _analyze_chain(chain.calls, chain.puts, spot, exp, rate, strikes_around=n)
        return _out({
            "ticker": ticker.upper(), "underlying_price": spot,
            "risk_free_rate_pct": round(rate * 100, 2),
            "summary": summary, **rows,
            "available_expirations": expirations[:16],
            "note": OPTIONS_NOTE,
            "source": f"Yahoo Finance via yfinance (https://finance.yahoo.com/quote/{ticker}/options)",
        })
    except Exception as e:
        return _err("yfinance options", e)


@tool
def get_options_overview(ticker: str, num_expirations: int = 4) -> str:
    """Options market overview across the next several expirations (term structure) for a
    stock, ETF or index. For each expiration: at-the-money implied volatility, expected move,
    put/call ratios, max pain and 25-delta skew. Use it to see whether options price a big move
    (e.g. around earnings), how fear (put demand) compares with greed, and how IV changes with time.
    num_expirations: 1-8 (each one is a separate data request)."""
    try:
        n = max(1, min(int(num_expirations), 8))
        check_and_record("yfinance")  # expiration list
        tk = yf.Ticker(ticker)
        expirations = list(tk.options or [])
        if not expirations:
            return _out({"error": f"No listed options found for '{ticker}'."})
        chosen = expirations[:n]
        rate = _risk_free_rate()
        spot, term = None, []
        for exp in chosen:
            check_and_record("yfinance")
            chain = tk.option_chain(exp)
            if chain.calls is None or chain.calls.empty:
                continue
            spot = spot or _spot_from_chain(chain, tk)
            if not spot:
                return _out({"error": f"Couldn't get the current price of {ticker}."})
            summary, _ = _analyze_chain(chain.calls, chain.puts, spot, exp, rate)
            term.append(summary)
        if not term:
            return _out({"error": f"No option quotes for {ticker}."})
        ivs = [s["atm_iv_pct"] for s in term if s["atm_iv_pct"]]
        shape = None
        if len(ivs) >= 2:
            shape = ("upward (normal: later dates cost more)" if ivs[-1] > ivs[0] + 1 else
                     "inverted (near-term fear or an event soon, e.g. earnings)" if ivs[0] > ivs[-1] + 1
                     else "flat")
        return _out({
            "ticker": ticker.upper(), "underlying_price": spot,
            "iv_term_structure": shape, "expirations": term,
            "more_expirations_available": expirations[n:n + 12],
            "note": OPTIONS_NOTE,
            "source": f"Yahoo Finance via yfinance (https://finance.yahoo.com/quote/{ticker}/options)",
        })
    except Exception as e:
        return _err("yfinance options", e)


@tool
def option_calculator(spot: float, strike: float, days_to_expiry: float, iv_percent: float,
                      option_type: str = "call", rate_percent: float | None = None,
                      dividend_yield_percent: float = 0.0) -> str:
    """Black-Scholes calculator for a single option (European style; a close estimate for
    US stock options). Inputs: underlying price, strike, days to expiry, implied volatility
    in percent (e.g. 30), option_type 'call' or 'put'. rate_percent defaults to the current
    13-week T-bill yield. Returns theoretical price, delta, gamma, theta per day, vega,
    rho, probability of expiring in the money, and the breakeven price at expiry.
    Use it for 'what if' questions, e.g. how a price change or IV drop affects an option."""
    try:
        kind = option_type.lower().strip()
        if kind not in ("call", "put"):
            return _out({"error": "option_type must be 'call' or 'put'."})
        rate = rate_percent / 100 if rate_percent is not None else _risk_free_rate()
        g = black_scholes(float(spot), float(strike), float(days_to_expiry) / 365,
                          float(iv_percent) / 100, rate, kind, float(dividend_yield_percent) / 100)
        if not g:
            return _out({"error": "spot, strike, days_to_expiry and iv_percent must all be > 0."})
        price = g["price"]
        return _out({
            "inputs": {"spot": spot, "strike": strike, "days_to_expiry": days_to_expiry,
                       "iv_pct": iv_percent, "type": kind, "rate_pct": round(rate * 100, 2),
                       "dividend_yield_pct": dividend_yield_percent},
            "theoretical_price": round(price, 4),
            "delta": round(g["delta"], 4), "gamma": round(g["gamma"], 5),
            "theta_per_day": round(g["theta_per_day"], 4),
            "vega_per_1pt_iv": round(g["vega_per_1pt"], 4),
            "rho_per_1pt_rate": round(g["rho_per_1pt"], 4),
            "prob_expire_itm_pct": round(g["prob_itm"] * 100, 1),
            "breakeven_at_expiry": round(strike + price if kind == "call" else strike - price, 2),
            "note": "Model estimate (risk-neutral probabilities), not a market quote.",
            "source": "Calculated locally with the Black-Scholes model",
        })
    except Exception as e:
        return _err("option calculator", e)


# ==========================================================================
# Futures, FX and volatility snapshot
# ==========================================================================

SNAPSHOT_GROUPS = {
    "equity_index_futures": {"ES=F": "S&P 500 E-mini", "NQ=F": "Nasdaq 100 E-mini",
                             "YM=F": "Dow E-mini", "RTY=F": "Russell 2000 E-mini"},
    "energy": {"CL=F": "WTI crude oil", "BZ=F": "Brent crude oil", "NG=F": "Natural gas",
               "RB=F": "RBOB gasoline", "HO=F": "Heating oil"},
    "metals": {"GC=F": "Gold", "SI=F": "Silver", "HG=F": "Copper",
               "PL=F": "Platinum", "PA=F": "Palladium"},
    "agriculture": {"ZC=F": "Corn", "ZS=F": "Soybeans", "ZW=F": "Wheat", "KC=F": "Coffee",
                    "SB=F": "Sugar", "CT=F": "Cotton", "LE=F": "Live cattle"},
    "rates": {"ZT=F": "2-yr T-note futures", "ZF=F": "5-yr T-note futures",
              "ZN=F": "10-yr T-note futures", "ZB=F": "30-yr T-bond futures",
              "^IRX": "13-week T-bill yield (%)", "^FVX": "5-yr Treasury yield (%)",
              "^TNX": "10-yr Treasury yield (%)", "^TYX": "30-yr Treasury yield (%)"},
    "currency_futures": {"6E=F": "Euro", "6J=F": "Japanese yen", "6B=F": "British pound",
                         "6C=F": "Canadian dollar", "6A=F": "Australian dollar",
                         "6S=F": "Swiss franc", "DX=F": "US dollar index"},
    "fx": {"EURUSD=X": "EUR/USD", "USDJPY=X": "USD/JPY", "GBPUSD=X": "GBP/USD",
           "USDCHF=X": "USD/CHF", "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD",
           "USDCNY=X": "USD/CNY", "USDINR=X": "USD/INR", "DX-Y.NYB": "US dollar index (DXY)"},
    "volatility": {"^VIX": "VIX (S&P 500 30-day implied vol)", "^VIX9D": "VIX 9-day",
                   "^VIX3M": "VIX 3-month", "^VVIX": "VVIX (volatility of VIX)",
                   "^OVX": "Oil volatility (OVX)", "^GVZ": "Gold volatility (GVZ)"},
}


@tool
def get_market_snapshot(category: str, symbols: list[str] | None = None) -> str:
    """Prices and recent performance (1 day, 1 week, 1 month, 3 months) for a group of
    futures, currencies or volatility indices. category: one of equity_index_futures, energy,
    metals, agriculture, rates, currency_futures, fx, volatility; or 'custom' with your own
    Yahoo symbols in `symbols` (max 10, e.g. ['CL=F','GC=F','EURUSD=X']).
    Futures are the front-month continuous contract. The volatility category also reports
    the VIX term structure (contango = calm, backwardation = stress)."""
    try:
        cat = category.lower().strip()
        if cat == "custom":
            if not symbols:
                return _out({"error": "Pass `symbols` with category='custom'."})
            group = {s: s for s in symbols[:10]}
        elif cat in SNAPSHOT_GROUPS:
            group = SNAPSHOT_GROUPS[cat]
        else:
            return _out({"error": f"Unknown category '{category}'.",
                         "valid": list(SNAPSHOT_GROUPS) + ["custom"]})
        check_and_record("yfinance", n=len(group))
        df = yf.download(list(group), period="3mo", interval="1d", group_by="ticker",
                         auto_adjust=False, progress=False, threads=True)
        rows = []
        for sym, name in group.items():
            try:
                close = df[sym]["Close"].dropna()
            except KeyError:
                close = None
            if close is None or close.empty:
                rows.append({"symbol": sym, "name": name, "error": "no data"})
                continue
            last = float(close.iloc[-1])
            chg = lambda n: round((last / float(close.iloc[-1 - n]) - 1) * 100, 2) \
                if len(close) > n else None
            rows.append({"symbol": sym, "name": name, "last": round(last, 4),
                         "as_of": str(close.index[-1].date()),
                         "chg_1d_pct": chg(1), "chg_1w_pct": chg(5),
                         "chg_1m_pct": chg(21), "chg_3m_pct": chg(len(close) - 1)})
        out = {"category": cat, "instruments": rows,
               "source": "Yahoo Finance via yfinance (futures = front-month continuous contract)"}
        if cat == "volatility":
            v = {r["symbol"]: r.get("last") for r in rows}
            if v.get("^VIX9D") and v.get("^VIX") and v.get("^VIX3M"):
                out["vix_term_structure"] = (
                    "contango (VIX9D < VIX < VIX3M): markets calm" if v["^VIX9D"] < v["^VIX"] < v["^VIX3M"]
                    else "backwardation (short-term vol above longer-term): stress or event risk"
                    if v["^VIX9D"] > v["^VIX"] > v["^VIX3M"] else "mixed")
        return _out(out)
    except Exception as e:
        return _err("yfinance", e)


# ==========================================================================
# CFTC Commitments of Traders (futures positioning)
# ==========================================================================

CFTC_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"  # Legacy, futures only

# Verified contract codes (CFTC data, Oct 2026). Aliases -> cftc_contract_market_code.
COT_MARKETS = {
    "088691": ["gold", "gc"], "084691": ["silver", "si"], "085692": ["copper", "hg"],
    "076651": ["platinum", "pl"],
    "067651": ["crude", "crude oil", "wti", "oil", "cl"], "06765T": ["brent"],
    "023651": ["natural gas", "nat gas", "ng"], "111659": ["gasoline", "rbob", "rb"],
    "022651": ["heating oil", "ulsd", "diesel", "ho"],
    "002602": ["corn", "zc"], "005602": ["soybeans", "soy", "zs"], "001602": ["wheat", "zw"],
    "083731": ["coffee", "kc"], "080732": ["sugar", "sb"], "033661": ["cotton", "ct"],
    "057642": ["live cattle", "cattle", "le"],
    "13874A": ["s&p 500", "sp500", "s&p", "spx", "es", "e-mini s&p"],
    "209742": ["nasdaq", "nasdaq 100", "nq"], "124603": ["dow", "djia", "ym"],
    "239742": ["russell", "russell 2000", "rty"],
    "042601": ["2y", "2-year", "2 year note", "zt"], "044601": ["5y", "5-year", "zf"],
    "043602": ["10y", "10-year", "10 year note", "treasury", "zn"],
    "020601": ["30y", "bond", "t-bond", "zb"], "020604": ["ultra bond", "ub"],
    "134741": ["sofr", "sofr 3m"],
    "099741": ["euro", "eur", "6e"], "097741": ["yen", "jpy", "6j"],
    "096742": ["pound", "gbp", "sterling", "6b"], "090741": ["canadian dollar", "cad", "6c"],
    "232741": ["australian dollar", "aud", "6a"], "092741": ["swiss franc", "chf", "6s"],
    "095741": ["mexican peso", "mxn", "peso"],
    "133741": ["bitcoin", "btc"], "133742": ["micro bitcoin"],
    "146021": ["ether", "ethereum", "eth"], "1170E1": ["vix", "vx"],
}
_ALIAS_TO_CODE = {a: code for code, aliases in COT_MARKETS.items() for a in aliases}


def _cftc(params: dict) -> list:
    check_and_record("cftc")
    r = requests.get(CFTC_URL, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


@tool
def get_cot_report(market: str, weeks: int = 8) -> str:
    """CFTC Commitments of Traders: how different trader groups are positioned in a futures
    market. market: a name like 'gold', 'crude oil', 'S&P 500', '10y', 'euro', 'bitcoin', 'vix',
    'corn', or a CFTC contract code. Returns weekly long/short/net positions for large
    speculators (non-commercial, e.g. hedge funds), commercials (hedgers: producers and users)
    and small traders, plus open interest, week-over-week changes, and where speculators'
    net position sits in its 1-year range (0% = most bearish, 100% = most bullish in a year).
    Data is as of each Tuesday, published Fridays."""
    try:
        key = market.lower().strip()
        code = _ALIAS_TO_CODE.get(key) or (market.upper() if market.upper() in COT_MARKETS else None)
        candidates = None
        if not code:  # free-text search on the latest report, biggest market first
            latest = _cftc({"$select": "max(report_date_as_yyyy_mm_dd) as d"})[0]["d"]
            found = _cftc({
                "$select": "cftc_contract_market_code,market_and_exchange_names,open_interest_all",
                "$where": f"report_date_as_yyyy_mm_dd='{latest}' AND "
                          f"upper(market_and_exchange_names) like '%{market.upper().replace(chr(39), '')}%'",
                "$limit": 50,
            })
            if not found:
                return _out({"error": f"No CFTC futures market matches '{market}'.",
                             "try": sorted({a[0] for a in COT_MARKETS.values()})})
            found.sort(key=lambda r: -int(r.get("open_interest_all") or 0))
            code = found[0]["cftc_contract_market_code"]
            candidates = [r["market_and_exchange_names"] for r in found[:6]]

        rows = _cftc({"cftc_contract_market_code": code,
                      "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": 52})
        if not rows:
            return _out({"error": f"No COT data for contract code {code}."})
        i = lambda r, k: int(float(r.get(k) or 0))
        spec_net = [i(r, "noncomm_positions_long_all") - i(r, "noncomm_positions_short_all") for r in rows]
        lo, hi = min(spec_net), max(spec_net)
        weekly = []
        for r, sn in list(zip(rows, spec_net))[: max(1, min(int(weeks), 26))]:
            weekly.append({
                "date": r["report_date_as_yyyy_mm_dd"][:10],
                "open_interest": i(r, "open_interest_all"),
                "speculators_long": i(r, "noncomm_positions_long_all"),
                "speculators_short": i(r, "noncomm_positions_short_all"),
                "speculators_net": sn,
                "commercials_net": i(r, "comm_positions_long_all") - i(r, "comm_positions_short_all"),
                "small_traders_net": i(r, "nonrept_positions_long_all") - i(r, "nonrept_positions_short_all"),
                "chg_open_interest": i(r, "change_in_open_interest_all"),
                "chg_speculators_net": i(r, "change_in_noncomm_long_all") - i(r, "change_in_noncomm_short_all"),
            })
        latest_row = rows[0]
        out = {
            "market": latest_row["market_and_exchange_names"], "contract_code": code,
            "units": latest_row.get("contract_units"),
            "latest": weekly[0],
            "speculators_pct_of_oi": {"long": latest_row.get("pct_of_oi_noncomm_long_all"),
                                      "short": latest_row.get("pct_of_oi_noncomm_short_all")},
            "speculators_net_1y_range": {"low": lo, "high": hi,
                                         "position_pct": round((spec_net[0] - lo) / (hi - lo) * 100)
                                         if hi > lo else None, "weeks_of_data": len(rows)},
            "weekly": weekly,
            "note": "Legacy futures-only report. Extreme speculator positioning is often watched "
                    "as a contrarian signal, but it is not a timing tool.",
            "source": "CFTC Commitments of Traders (https://www.cftc.gov/MarketReports/CommitmentsofTraders)",
        }
        if candidates and len(candidates) > 1:
            out["other_matches"] = candidates[1:]
        return _out(out)
    except Exception as e:
        return _err("CFTC", e)


DERIVATIVE_TOOLS = [get_options_chain, get_options_overview, option_calculator,
                    get_market_snapshot, get_cot_report]
