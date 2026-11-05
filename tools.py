"""
Data tools the agent can call.

Every tool returns a compact JSON string. Keeping outputs small matters:
everything a tool returns is sent back to Claude, so big outputs cost tokens
and fill up the context window.

Sources (all free):
  - yfinance      : stocks, ETFs, indices, FX, futures (no key; unofficial, can break)
  - CoinGecko     : crypto (free Demo key recommended)
  - Finnhub       : company news (free key)
  - SEC EDGAR     : US company filings (no key, needs a User-Agent with your email)
  - FRED          : macro data (free key)
  - DuckDuckGo    : web + news search via the ddgs package (no key, no account)
  - CFTC          : futures positioning reports (no key; see derivatives.py)

Every request is counted and capped by usage.py, so free-plan limits are never exceeded.
"""

import os
from datetime import date, timedelta

import requests
import yfinance as yf
from langchain_core.tools import tool

from common import HTTP_TIMEOUT, _err, _out
from derivatives import DERIVATIVE_TOOLS
from usage import check_and_record


# --------------------------------------------------------------------------
# Stocks & ETFs (yfinance)
# --------------------------------------------------------------------------

@tool
def get_quote(ticker: str) -> str:
    """Current price and key stats for a stock, ETF, index, currency pair or future.
    Use Yahoo Finance symbols, e.g. AAPL, SPY, ^GSPC (S&P 500), ^IXIC, EURUSD=X, GC=F (gold), BTC-USD.
    For ETFs it also returns expense ratio, total assets and yield when available."""
    try:
        check_and_record("yfinance")
        info = yf.Ticker(ticker).info or {}
        if not info or info.get("quoteType") is None:
            return _out({"error": f"No data for '{ticker}'. Check the symbol."})
        keys = [
            "symbol", "shortName", "quoteType", "currency", "exchange",
            "currentPrice", "regularMarketPrice", "previousClose",
            "regularMarketChangePercent", "fiftyTwoWeekLow", "fiftyTwoWeekHigh",
            "fiftyDayAverage", "twoHundredDayAverage", "marketCap", "volume",
            "averageVolume", "trailingPE", "forwardPE", "priceToBook",
            "dividendYield", "beta", "sector", "industry",
            # ETF / fund fields
            "category", "fundFamily", "totalAssets", "netExpenseRatio",
            "annualReportExpenseRatio", "yield", "ytdReturn",
            "threeYearAverageReturn", "fiveYearAverageReturn",
            # Analyst fields
            "targetMeanPrice", "recommendationKey", "numberOfAnalystOpinions",
        ]
        data = {k: info[k] for k in keys if info.get(k) is not None}
        data["source"] = f"Yahoo Finance via yfinance (https://finance.yahoo.com/quote/{ticker})"
        return _out(data)
    except Exception as e:
        return _err("yfinance", e)


@tool
def get_price_history(ticker: str, period: str = "1y") -> str:
    """Price performance summary over a period for any Yahoo Finance symbol.
    period: one of 5d, 1mo, 3mo, 6mo, ytd, 1y, 2y, 5y, 10y, max.
    Returns start/end price, total return, high/low, annualized volatility,
    max drawdown, and a few sampled closing prices (not the full series)."""
    try:
        check_and_record("yfinance")
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist.empty:
            return _out({"error": f"No price history for '{ticker}' over {period}."})
        close = hist["Close"]
        rets = close.pct_change().dropna()
        drawdown = (close / close.cummax() - 1).min()
        step = max(1, len(close) // 12)
        samples = {str(i.date()): round(float(v), 2) for i, v in close.iloc[::step].items()}
        return _out({
            "ticker": ticker,
            "period": period,
            "start": str(close.index[0].date()),
            "end": str(close.index[-1].date()),
            "start_price": round(float(close.iloc[0]), 2),
            "end_price": round(float(close.iloc[-1]), 2),
            "total_return_pct": round(float((close.iloc[-1] / close.iloc[0] - 1) * 100), 2),
            "period_high": round(float(close.max()), 2),
            "period_low": round(float(close.min()), 2),
            "annualized_volatility_pct": round(float(rets.std() * (252 ** 0.5) * 100), 2),
            "max_drawdown_pct": round(float(drawdown * 100), 2),
            "sampled_closes": samples,
            "source": "Yahoo Finance via yfinance (dividend/split adjusted)",
        })
    except Exception as e:
        return _err("yfinance", e)


@tool
def get_fundamentals(ticker: str) -> str:
    """Key financials for a company from its last 4 annual reports:
    revenue, gross profit, operating income, net income, EPS, free cash flow,
    total debt, cash. Use for valuation and growth questions about stocks."""
    try:
        check_and_record("yfinance", n=4)  # info + 3 financial statements
        t = yf.Ticker(ticker)
        rows = {
            "income": (t.income_stmt, ["Total Revenue", "Gross Profit", "Operating Income",
                                        "Net Income", "Diluted EPS"]),
            "cashflow": (t.cashflow, ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"]),
            "balance": (t.balance_sheet, ["Total Debt", "Cash And Cash Equivalents",
                                          "Stockholders Equity"]),
        }
        out = {"ticker": ticker, "currency": (t.info or {}).get("financialCurrency")}
        for name, (df, wanted) in rows.items():
            if df is None or df.empty:
                continue
            section = {}
            for row in wanted:
                if row in df.index:
                    section[row] = {str(c.date()): (None if v != v else float(v))
                                    for c, v in df.loc[row].items()}
            out[name] = section
        out["source"] = "Company filings via Yahoo Finance (yfinance)"
        return _out(out)
    except Exception as e:
        return _err("yfinance", e)


@tool
def get_etf_holdings(ticker: str) -> str:
    """Top holdings and sector weights of an ETF or mutual fund, e.g. SPY, QQQ, VTI, ARKK."""
    try:
        check_and_record("yfinance")
        fd = yf.Ticker(ticker).funds_data
        out = {"ticker": ticker}
        th = fd.top_holdings
        if th is not None and not th.empty:
            out["top_holdings"] = th.reset_index().head(15).to_dict(orient="records")
        sw = fd.sector_weightings
        if sw:
            out["sector_weightings"] = sw
        ac = fd.asset_classes
        if ac:
            out["asset_classes"] = ac
        if len(out) == 1:
            return _out({"error": f"No fund data for '{ticker}'. Is it an ETF/fund?"})
        out["source"] = "Yahoo Finance via yfinance"
        return _out(out)
    except Exception as e:
        return _err("yfinance", e)


# --------------------------------------------------------------------------
# Crypto (CoinGecko)
# --------------------------------------------------------------------------

def _coingecko(path: str, params: dict | None = None) -> dict:
    check_and_record("coingecko")  # each request uses one monthly credit
    headers = {"accept": "application/json"}
    key = os.getenv("COINGECKO_API_KEY")
    if key:
        headers["x-cg-demo-api-key"] = key
    r = requests.get(f"https://api.coingecko.com/api/v3{path}", params=params,
                     headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


@tool
def get_crypto_data(coin: str) -> str:
    """Market data for a cryptocurrency by name or symbol (e.g. 'bitcoin', 'ETH', 'solana').
    Returns price, market cap & rank, volume, % change over 24h/7d/30d/1y,
    all-time high, supply figures and a short description."""
    try:
        found = _coingecko("/search", {"query": coin}).get("coins", [])
        if not found:
            return _out({"error": f"No coin found for '{coin}'."})
        coin_id = found[0]["id"]
        d = _coingecko(f"/coins/{coin_id}", {
            "localization": "false", "tickers": "false", "community_data": "false",
            "developer_data": "false", "sparkline": "false",
        })
        m = d.get("market_data", {})
        usd = lambda k: (m.get(k) or {}).get("usd")
        return _out({
            "id": coin_id, "name": d.get("name"), "symbol": d.get("symbol"),
            "market_cap_rank": d.get("market_cap_rank"),
            "price_usd": usd("current_price"),
            "market_cap_usd": usd("market_cap"),
            "volume_24h_usd": usd("total_volume"),
            "change_24h_pct": m.get("price_change_percentage_24h"),
            "change_7d_pct": m.get("price_change_percentage_7d"),
            "change_30d_pct": m.get("price_change_percentage_30d"),
            "change_1y_pct": m.get("price_change_percentage_1y"),
            "ath_usd": usd("ath"), "ath_date": (m.get("ath_date") or {}).get("usd"),
            "circulating_supply": m.get("circulating_supply"),
            "max_supply": m.get("max_supply"),
            "categories": d.get("categories", [])[:5],
            "description": (d.get("description", {}).get("en") or "")[:600],
            "last_updated": d.get("last_updated"),
            "source": f"CoinGecko (https://www.coingecko.com/en/coins/{coin_id})",
        })
    except Exception as e:
        return _err("CoinGecko", e)


@tool
def get_crypto_market_overview() -> str:
    """Overall crypto market snapshot: total market cap, 24h volume, BTC/ETH dominance,
    plus the top 10 coins by market cap and currently trending coins."""
    try:
        g = _coingecko("/global").get("data", {})
        top = _coingecko("/coins/markets", {"vs_currency": "usd", "order": "market_cap_desc",
                                            "per_page": 10, "page": 1})
        trending = _coingecko("/search/trending").get("coins", [])
        return _out({
            "total_market_cap_usd": g.get("total_market_cap", {}).get("usd"),
            "total_volume_24h_usd": g.get("total_volume", {}).get("usd"),
            "market_cap_change_24h_pct": g.get("market_cap_change_percentage_24h_usd"),
            "dominance_pct": {k: round(v, 2) for k, v in
                              list(g.get("market_cap_percentage", {}).items())[:5]},
            "top_10": [{"name": c["name"], "symbol": c["symbol"], "price": c["current_price"],
                        "market_cap": c["market_cap"],
                        "change_24h_pct": c.get("price_change_percentage_24h")} for c in top],
            "trending": [c["item"]["name"] for c in trending[:7]],
            "source": "CoinGecko (https://www.coingecko.com)",
        })
    except Exception as e:
        return _err("CoinGecko", e)


# --------------------------------------------------------------------------
# News (Finnhub) and web search (DuckDuckGo)
# --------------------------------------------------------------------------

@tool
def get_company_news(ticker: str, days: int = 7) -> str:
    """Recent news headlines about a US-listed company from Finnhub (last N days, max 30)."""
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        return _out({"error": "FINNHUB_API_KEY not set. Use web_search for news instead."})
    try:
        days = max(1, min(int(days), 30))
        check_and_record("finnhub")
        r = requests.get("https://finnhub.io/api/v1/company-news", timeout=HTTP_TIMEOUT, params={
            "symbol": ticker.upper(),
            "from": str(date.today() - timedelta(days=days)),
            "to": str(date.today()),
            "token": key,
        })
        r.raise_for_status()
        items = r.json()[:12]
        return _out({"ticker": ticker, "articles": [{
            "headline": a.get("headline"), "source": a.get("source"),
            "date": date.fromtimestamp(a["datetime"]).isoformat() if a.get("datetime") else None,
            "summary": (a.get("summary") or "")[:300], "url": a.get("url"),
        } for a in items]})
    except Exception as e:
        return _err("Finnhub", e)


def _timelimit(days: int | None) -> str | None:
    """Map a number of days to DuckDuckGo's time filter: d, w, m or y."""
    if not days:
        return None
    days = int(days)
    return "d" if days <= 1 else "w" if days <= 7 else "m" if days <= 31 else "y"


@tool
def web_search(query: str, recent_days: int | None = None) -> str:
    """Search the web for news, analysis, regulation, macro events, or anything the
    structured data tools don't cover. Set recent_days (e.g. 7) to search recent news
    articles only. Returns titles, URLs and snippets; cite the URLs in your answer.
    Snippets are short: they summarize a page, they don't contain the full article."""
    try:
        from ddgs import DDGS
        check_and_record("ddgs")
        tl = _timelimit(recent_days)
        with DDGS() as ddg:
            if tl:  # news search, filtered by date
                raw = ddg.news(query, region="us-en", timelimit=tl, max_results=8)
                results = [{
                    "title": r.get("title"), "url": r.get("url") or r.get("href"),
                    "source": r.get("source"), "published": r.get("date"),
                    "content": (r.get("body") or "")[:500],
                } for r in raw]
            else:   # general web search
                raw = ddg.text(query, region="us-en", max_results=8)
                results = [{
                    "title": r.get("title"), "url": r.get("href") or r.get("url"),
                    "content": (r.get("body") or "")[:500],
                } for r in raw]
        if not results:
            return _out({"query": query, "results": [],
                         "note": "No results. Try a broader or differently worded query."})
        return _out({"query": query, "results": results, "source": "DuckDuckGo search"})
    except Exception as e:
        return _err("DuckDuckGo search", e)


# --------------------------------------------------------------------------
# SEC filings (EDGAR) and macro data (FRED)
# --------------------------------------------------------------------------

_SEC_HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT", "MarketResearchAgent contact@example.com")}
_cik_cache: dict[str, str] = {}


@tool
def get_sec_filings(ticker: str, form_type: str = "") -> str:
    """Latest SEC filings for a US company (10-K annual, 10-Q quarterly, 8-K events,
    S-1 IPO, 13F, Form 4 insider trades). Optionally filter with form_type, e.g. '10-K'.
    Returns filing dates and direct links to the documents."""
    try:
        if not _cik_cache:
            check_and_record("sec")
            r = requests.get("https://www.sec.gov/files/company_tickers.json",
                             headers=_SEC_HEADERS, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            for row in r.json().values():
                _cik_cache[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
        cik = _cik_cache.get(ticker.upper())
        if not cik:
            return _out({"error": f"'{ticker}' not found in SEC EDGAR (US filers only)."})
        check_and_record("sec")
        r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                         headers=_SEC_HEADERS, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        sub = r.json()
        recent = sub["filings"]["recent"]
        filings = []
        for i, form in enumerate(recent["form"]):
            if form_type and form.upper() != form_type.upper():
                continue
            acc = recent["accessionNumber"][i].replace("-", "")
            filings.append({
                "form": form, "filed": recent["filingDate"][i],
                "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/"
                       f"{recent['primaryDocument'][i]}",
            })
            if len(filings) >= 10:
                break
        return _out({"company": sub.get("name"), "cik": cik, "filings": filings,
                     "source": "SEC EDGAR"})
    except Exception as e:
        return _err("SEC EDGAR", e)


@tool
def get_economic_data(series_id: str) -> str:
    """Latest values of a US macroeconomic series from FRED. Common series IDs:
    FEDFUNDS (Fed funds rate), CPIAUCSL (CPI), UNRATE (unemployment), GDP,
    DGS10 (10-yr Treasury yield), T10Y2Y (yield curve), M2SL (money supply),
    VIXCLS (VIX), DCOILWTICO (WTI oil), MORTGAGE30US."""
    key = os.getenv("FRED_API_KEY")
    if not key:
        return _out({"error": "FRED_API_KEY not set."})
    try:
        base = "https://api.stlouisfed.org/fred"
        p = {"series_id": series_id.upper(), "api_key": key, "file_type": "json"}
        check_and_record("fred", n=2)  # series info + observations
        meta = requests.get(f"{base}/series", params=p, timeout=HTTP_TIMEOUT)
        meta.raise_for_status()
        obs = requests.get(f"{base}/series/observations", timeout=HTTP_TIMEOUT,
                           params={**p, "sort_order": "desc", "limit": 13})
        obs.raise_for_status()
        s = meta.json()["seriess"][0]
        return _out({
            "series": s["title"], "units": s["units"], "frequency": s["frequency"],
            "latest_observations": [{"date": o["date"], "value": o["value"]}
                                    for o in obs.json()["observations"]],
            "source": f"FRED (https://fred.stlouisfed.org/series/{series_id.upper()})",
        })
    except Exception as e:
        return _err("FRED", e)


ALL_TOOLS = [
    get_quote, get_price_history, get_fundamentals, get_etf_holdings,
    get_crypto_data, get_crypto_market_overview,
    get_company_news, web_search, get_sec_filings, get_economic_data,
] + DERIVATIVE_TOOLS  # options, futures, FX, volatility, COT (see derivatives.py)
