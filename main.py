"""
SGG Evergreen Monitor — FastAPI backend.

All metrics (TRI, CAGR, trailing returns, std dev, drawdown, downside capture,
NAV-weighted portfolio returns) are computed from the raw Excel on EVERY request
— no caching — so spreadsheet edits show up immediately.

Benchmark handling is graceful: if a `pitchbook_tri_index` sheet (with a
`pitchbook_index` column) exists it is used as the benchmark; if not, the
benchmark line is omitted and downside-capture returns null. Nothing crashes.
"""

import os
import math

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

EXCEL_PATH = os.environ.get("EXCEL_PATH", "data/06_10_26_Evergreen_Database_v11.xlsx")
PERF_SHEET = "fund_performance"
META_SHEET = "fund_metadata"
# v11 renamed the benchmark sheet from "pitchbook_tri_index" -> "pitchbook_index"
# (the column inside is still "pitchbook_index"). PB_SHEET drives the single
# benchmark used for downside-capture (Fund Details + risk metrics).
PB_SHEET = "pitchbook_index"

# All benchmark indices overlaid on the Normalized Total Returns chart. Each is
# loaded, collapsed to month-end (daily series get resampled), and normalized to
# base 100 on the front end. `key` drives the line color; `label` is the legend
# text. The loader still falls back to the lone non-date column if `col` is ever
# missing, so a future header change won't break the line.
BENCHMARKS = [
    {"key": "pitchbook",    "sheet": "pitchbook_index",    "col": "pitchbook_index",    "label": "Pitchbook Morningstar Evergreen"},
    {"key": "stepstone_pm", "sheet": "stepstone_pm_index", "col": "stepstone_pm_index", "label": "StepStone Private Markets"},
    {"key": "sp500",        "sheet": "sp500_index",        "col": "sp500_index",        "label": "S&P 500"},
]

# Original purchase amounts by fund_id. Used as the fallback beginning-of-month
# NAV for a fund's first held month, when there is no prior-month sgg_nav.
# This is the ONLY hardcoded data in the portfolio calculation — everything else
# reads dynamically from the workbook.
PURCHASE_AMOUNTS = {
    1: 5000000,    # Bow River
    2: 5000000,    # Carlyle AlpInvest
    3: 10000000,   # Cliffwater
    4: 10000000,   # Coller
    5: 5000000,    # StepStone PC Income
    6: 10000000,   # StepStone Private Markets
    7: 5000000,    # StepStone PV/Growth
    8: 10000000,   # AMG Pantheon
    9: 10000000,   # FLEX
    10: 10000000,  # Hamilton Lane
    11: 10000000,  # JPMF
}

# Fund logo file base names (extension auto-detected). NOTE: fund 2 uses the
# actual on-disk spelling "carlyle_alphinvest" (the marketing file is spelled
# with an 'h'); everything else matches the supplied mapping.
LOGO_BASENAMES = {
    1: "Bow_River_Capital_Logo",
    2: "carlyle_alphinvest",
    3: "cliffwater",
    4: "coller_capital_logo",
    5: "stepstone",
    6: "stepstone",
    7: "stepstone",
    8: "amg_pantheon",
    9: "flex",
    10: "hamilton_lane",
    11: "jpmf",
}
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".svg", ".webp"]

app = FastAPI(title="SGG Evergreen Monitor")


# ─────────────────────────────────────────── data loading (fresh each call) ──
def _read(sheet):
    return pd.read_excel(EXCEL_PATH, sheet_name=sheet, engine="openpyxl")


def load_perf():
    df = _read(PERF_SHEET)
    df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
    df = df[df["month_end"].notna()].copy()
    return df.sort_values(["fund_id", "month_end"])


def load_meta():
    df = _read(META_SHEET)
    df["class_inception_date"] = pd.to_datetime(
        df["class_inception_date"], errors="coerce"
    )
    return df


def load_benchmark():
    """Return DataFrame[month_end, pitchbook_index] sorted, or None if absent."""
    try:
        xl = pd.ExcelFile(EXCEL_PATH, engine="openpyxl")
        if PB_SHEET not in xl.sheet_names:
            return None
        pb = xl.parse(PB_SHEET)
        if "month_end" not in pb.columns or "pitchbook_index" not in pb.columns:
            return None
        pb["month_end"] = pd.to_datetime(pb["month_end"], errors="coerce")
        pb = pb[pb["month_end"].notna() & pb["pitchbook_index"].notna()]
        return pb[["month_end", "pitchbook_index"]].sort_values("month_end")
    except Exception:
        return None


def _load_index(sheet, col):
    """Load a benchmark index sheet as DataFrame[month_end, value], collapsed to
    one row per calendar month (last observation) and dated to month-end so daily
    series (e.g. StepStone) align exactly to the funds' monthly grid. Returns None
    if the sheet/columns are absent or empty."""
    try:
        xl = pd.ExcelFile(EXCEL_PATH, engine="openpyxl")
        if sheet not in xl.sheet_names:
            return None
        df = xl.parse(sheet)
        if "month_end" not in df.columns:
            return None
        if col not in df.columns:
            # Fall back to the only non-date column (guards the misspelled header).
            others = [c for c in df.columns if c != "month_end"]
            if not others:
                return None
            col = others[0]
        df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
        df = df[df["month_end"].notna() & df[col].notna()]
        if df.empty:
            return None
        per = df.sort_values("month_end").groupby(df["month_end"].dt.to_period("M")).last()
        per.index = per.index.to_timestamp(how="end").normalize()
        out = per[[col]].reset_index()
        out.columns = ["month_end", "value"]
        return out.sort_values("month_end")
    except Exception:
        return None


def load_benchmarks():
    """Return [{key,label,df}] for every configured benchmark present in the
    workbook (df = DataFrame[month_end, value]). Empty list if none present."""
    out = []
    for b in BENCHMARKS:
        df = _load_index(b["sheet"], b["col"])
        if df is not None and len(df) >= 1:
            out.append({"key": b["key"], "label": b["label"], "df": df})
    return out


# ──────────────────────────────────────────────────────────── helpers ──
def _num(x):
    """JSON-safe float (NaN/inf -> None)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _months_between(a, b):
    """Whole months from a to b (b >= a)."""
    return (b.year - a.year) * 12 + (b.month - a.month)


def _datestr(d):
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def fund_series(g):
    """
    Build an inception-anchored monthly series for one fund.

    Only reported monthly_return values are used to chain-link the index. NAV
    movements are NOT used to impute returns, and months without a reported
    monthly_return are not carried forward — the series stops at the last month
    with a real reported return.

    Returns dict: dates[], returns[] (effective monthly returns, None at base),
    tri[] (base 100 at inception), inception(Timestamp), or None if no data.
    """
    g = g.sort_values("month_end").reset_index(drop=True)
    nav = g["nav_per_share"]
    ret = g["monthly_return"]

    # Inception = first month with nav_per_share; fall back to first monthly_return.
    if nav.notna().any():
        start = int(nav.notna().idxmax())
    elif ret.notna().any():
        start = int(ret.notna().idxmax())
    else:
        return None

    dates, returns, tri = [], [], []
    cur = None
    for i in range(start, len(g)):
        if i == start:
            cur = 100.0
            r = None
        else:
            r = ret.iloc[i]
            if pd.isna(r):
                # No reported monthly return: do not impute from NAV and do not
                # carry forward. Reported returns are contiguous, so stop here.
                break
            r = float(r)
            cur = cur * (1.0 + r)
        dates.append(g["month_end"].iloc[i])
        returns.append(r)
        tri.append(cur)

    if not dates:
        return None

    return {
        "dates": dates,
        "returns": returns,
        "tri": tri,
        "inception": dates[0],
    }


def _product(rets):
    p = 1.0
    for r in rets:
        p *= 1.0 + r
    return p - 1.0


def max_drawdown(dates, tri):
    """Return drawdown stats dict from a TRI series."""
    out = {
        "max_drawdown": None,
        "peak_date": None,
        "trough_date": None,
        "time_to_trough": None,
        "recovery_date": None,
        "recovery_time": None,
    }
    pts = [(d, v) for d, v in zip(dates, tri) if v is not None]
    if len(pts) < 2:
        return out

    run_peak = pts[0][1]
    run_peak_date = pts[0][0]
    maxdd = 0.0
    trough_date = None
    peak_date = None
    peak_val = None
    for d, v in pts:
        if v > run_peak:
            run_peak = v
            run_peak_date = d
        dd = v / run_peak - 1.0
        if dd < maxdd:
            maxdd = dd
            trough_date = d
            peak_date = run_peak_date
            peak_val = run_peak

    if trough_date is None:
        return out  # monotonic, no drawdown

    recovery_date = None
    for d, v in pts:
        if d > trough_date and v >= peak_val:
            recovery_date = d
            break

    out["max_drawdown"] = _num(maxdd)
    out["peak_date"] = _datestr(peak_date)
    out["trough_date"] = _datestr(trough_date)
    out["time_to_trough"] = _months_between(peak_date, trough_date)
    if recovery_date is not None:
        out["recovery_date"] = _datestr(recovery_date)
        out["recovery_time"] = _months_between(trough_date, recovery_date)
    return out


def benchmark_returns(bench):
    """Map month_end(Timestamp) -> benchmark monthly return."""
    if bench is None or len(bench) < 2:
        return {}
    b = bench.sort_values("month_end").reset_index(drop=True)
    out = {}
    for i in range(1, len(b)):
        prev = b["pitchbook_index"].iloc[i - 1]
        cur = b["pitchbook_index"].iloc[i]
        if prev and prev != 0:
            out[pd.Timestamp(b["month_end"].iloc[i])] = float(cur / prev - 1.0)
    return out


def downside_capture(g, bret):
    """
    Downside capture for one fund's performance group `g` against the
    benchmark monthly-return map `bret` (month_end -> benchmark return).

    Uses the fund's RAW `monthly_return` column (not NAV-derived returns).
    Aligns by month_end on months where the fund has a non-null monthly_return
    AND the benchmark has a calculable return. Among benchmark-down months
    (benchmark return < 0):

        pct = mean(fund returns) / mean(benchmark returns) * 100

    Returns the pct rounded to 1 decimal as a float, or the string "n/a" when
    there are no benchmark-down months in the aligned history. Never None.
    """
    if "monthly_return" not in g.columns:
        return "n/a"
    fund_ret = {
        pd.Timestamp(row["month_end"]): float(row["monthly_return"])
        for _, row in g.iterrows()
        if pd.notna(row["month_end"]) and pd.notna(row["monthly_return"])
    }
    # Aligned history: months present in both series.
    down = [(fund_ret[m], bret[m]) for m in fund_ret if m in bret and bret[m] < 0]
    if not down:
        return "n/a"
    f_avg = sum(d[0] for d in down) / len(down)
    b_avg = sum(d[1] for d in down) / len(down)
    return round(f_avg / b_avg * 100.0, 1)


def _all_funds_cutoff(perf):
    """Most recent month_end where EVERY fund reports a monthly_return.

    This is the common as-of endpoint for the Risk/Return comparison tables, so
    every fund's metrics are measured to the exact same month (apples-to-apples).
    Auto-advances as lagging funds post. Falls back to the latest reported month
    if no single month has all funds (e.g. a permanently-redeemed fund)."""
    rep = perf[perf["monthly_return"].notna()]
    if rep.empty:
        return None
    total = perf["fund_id"].nunique()
    cnt = rep.groupby("month_end")["fund_id"].nunique()
    full = cnt[cnt >= total]
    if len(full):
        return pd.Timestamp(full.index.max())
    return pd.Timestamp(rep["month_end"].max())


def _index_series(df, cutoff=None):
    """Build {dates, returns, tri, inception} for a benchmark index df
    (month_end, value), optionally truncated to <= cutoff. Mirrors fund_series so
    the same metric helpers (drawdown, std, trailing, cagr) apply unchanged."""
    d = df.sort_values("month_end")
    if cutoff is not None:
        d = d[d["month_end"] <= cutoff]
    d = d.reset_index(drop=True)
    if len(d) < 1:
        return None
    vals = [float(v) for v in d["value"].tolist()]
    dates = list(d["month_end"])
    base = vals[0]
    if not base:
        return None
    tri = [v / base * 100.0 for v in vals]
    returns = [None] + [vals[i] / vals[i - 1] - 1.0 for i in range(1, len(vals))]
    return {"dates": dates, "returns": returns, "tri": tri, "inception": dates[0]}


def _downside_capture_map(ret_map, bret):
    """Downside capture from a month->return map against the benchmark map `bret`.
    Pitchbook vs Pitchbook returns exactly 100.0. 'n/a' if no benchmark-down months."""
    down = [(ret_map[m], bret[m]) for m in ret_map if m in bret and bret[m] < 0]
    if not down:
        return "n/a"
    f = sum(d[0] for d in down) / len(down)
    b = sum(d[1] for d in down) / len(down)
    return round(f / b * 100.0, 1)


def _index_metrics(cutoff, bret):
    """Risk + return metrics for each benchmark index, truncated to <= cutoff,
    using the SAME methodology as the fund tables. YTD uses the cutoff's calendar
    year (so an index with no data that year shows a dash). Each dict carries both
    risk and return fields; the two endpoints render the relevant subset."""
    ytd_year = cutoff.year if cutoff is not None else None
    out = []
    for bd in load_benchmarks():
        s = _index_series(bd["df"], cutoff)
        if s is None:
            out.append({"key": bd["key"], "fund_name": bd["label"]})
            continue
        dd = max_drawdown(s["dates"], s["tri"])
        rets_all = [r for r in s["returns"] if r is not None]
        std = _num(np.std(rets_all, ddof=1) * math.sqrt(12)) if len(rets_all) >= 2 else None
        rd = [(pd.Timestamp(d), r) for d, r in zip(s["dates"], s["returns"]) if r is not None]
        rets = [r for _, r in rd]
        tri = [v for v in s["tri"] if v is not None]
        latest = s["dates"][-1]
        msi = _months_between(s["inception"], latest)
        tri_final = tri[-1] if tri else None
        total_ret = _num(tri_final / 100.0 - 1.0) if tri_final else None
        cagr = _num((tri_final / 100.0) ** (12.0 / msi) - 1.0) if (tri_final and msi > 0) else None
        trailing_1y = _num(_product(rets[-12:])) if len(rets) >= 12 else None
        trailing_3y = _num(_product(rets[-36:])) if len(rets) >= 36 else None
        yr = ytd_year if ytd_year is not None else latest.year
        ytd_rets = [r for d, r in rd if d.year == yr]
        ytd = _num(_product(ytd_rets)) if ytd_rets else None
        dcr = _downside_capture_map({pd.Timestamp(d): r for d, r in rd}, bret)
        out.append({
            "key": bd["key"], "fund_name": bd["label"],
            "max_drawdown": dd["max_drawdown"], "time_to_trough": dd["time_to_trough"],
            "recovery_time": dd["recovery_time"], "annualized_std_dev": std,
            "downside_capture_ratio": dcr,
            "ytd_return": ytd, "trailing_1y": trailing_1y, "trailing_3y": trailing_3y,
            "annualized_since_inception": cagr, "total_return_since_inception": total_ret,
        })
    return out


# ─────────────────────────────────────────────────────────────── endpoints ──
@app.get("/api/funds")
def get_funds():
    meta = load_meta()

    def sval(r, c):
        """String/raw metadata value, or None if missing/blank."""
        if c not in meta.columns:
            return None
        v = r.get(c)
        if not pd.notna(v):
            return None
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v

    def fval(r, c):
        """Numeric metadata value (e.g. a fee like 0.0175), or None."""
        if c not in meta.columns:
            return None
        return _num(r.get(c)) if pd.notna(r.get(c)) else None

    rows = []
    for _, r in meta.iterrows():
        inc = None
        if "class_inception_date" in meta.columns and pd.notna(r.get("class_inception_date")):
            # ISO datetime so the front-end formatter renders it as "Jun 2020".
            inc = pd.Timestamp(r["class_inception_date"]).strftime("%Y-%m-%dT00:00:00")
        rows.append(
            {
                "fund_id": int(r["fund_id"]),
                "fund_name": sval(r, "fund_name"),
                "sec_name": sval(r, "sec_name"),
                "strategy": sval(r, "strategy"),
                "share_class": sval(r, "share_class"),
                "fund_type": sval(r, "fund_type"),
                "ticker": sval(r, "ticker"),
                "cik": sval(r, "cik"),
                "class_inception_date": inc,
                "management_fee": fval(r, "management_fee"),
                "expense_ratio": fval(r, "expense_ratio"),
                "incentive_fee": fval(r, "incentive_fee"),
                "gate": fval(r, "gate"),
                "website_link": sval(r, "website_link"),
            }
        )
    rows.sort(key=lambda x: (x["fund_name"] or "").lower())
    return {"count": len(rows), "funds": rows}


@app.get("/api/nav")
def get_nav():
    perf = load_perf()
    meta = load_meta()
    name = dict(zip(meta["fund_id"], meta["fund_name"]))
    strat = dict(zip(meta["fund_id"], meta["strategy"]))

    rows = []
    for fid, g in perf.groupby("fund_id"):
        g = g.sort_values("month_end")
        navrows = g[g["nav_per_share"].notna()]
        if len(navrows):
            last = navrows.iloc[-1]
            month_end = _datestr(last["month_end"])
            nav_ps = _num(last["nav_per_share"])
        else:
            month_end = None
            nav_ps = None
        sggrows = g[g["sgg_nav"].notna()]
        sgg = _num(sggrows.iloc[-1]["sgg_nav"]) if len(sggrows) else None
        rows.append(
            {
                "fund_id": int(fid),
                "fund_name": name.get(fid),
                "strategy": strat.get(fid),
                "month_end": month_end,
                "nav_per_share": nav_ps,
                "sgg_nav": sgg,
            }
        )
    rows.sort(key=lambda x: (x["fund_name"] or "").lower())
    return {"count": len(rows), "navs": rows}


def _all_fund_series(perf, meta):
    name = dict(zip(meta["fund_id"], meta["fund_name"]))
    out = {}
    for fid, g in perf.groupby("fund_id"):
        s = fund_series(g)
        if s is not None:
            out[int(fid)] = {"name": name.get(fid), **s}
    return out


@app.get("/api/tri")
def get_tri():
    perf = load_perf()
    meta = load_meta()
    benches = load_benchmarks()
    series = _all_fund_series(perf, meta)

    # Master date axis = union of all fund dates (+ every benchmark).
    all_dates = set()
    for s in series.values():
        all_dates.update(s["dates"])
    for bd in benches:
        all_dates.update(pd.Timestamp(d) for d in bd["df"]["month_end"])
    dates = sorted(all_dates)
    idx = {d: i for i, d in enumerate(dates)}

    out_series = []
    for fid, s in series.items():
        data = [None] * len(dates)
        for d, v in zip(s["dates"], s["tri"]):
            data[idx[d]] = _num(v)
        out_series.append({"fund_id": fid, "fund_name": s["name"], "data": data, "benchmark": False})

    # One dashed line per benchmark, normalized to base 100 at its own first point
    # (the front end rebases again to the funds' common start). bench_key drives
    # the line color in the UI.
    for bd in benches:
        df = bd["df"]
        base = df["value"].iloc[0]
        data = [None] * len(dates)
        if base and base != 0:
            for _, r in df.iterrows():
                data[idx[pd.Timestamp(r["month_end"])]] = _num(r["value"] / base * 100.0)
        out_series.append(
            {"fund_id": None, "fund_name": bd["label"], "data": data,
             "benchmark": True, "bench_key": bd["key"]}
        )

    return {
        "dates": [_datestr(d) for d in dates],
        "series": out_series,
        "benchmark_available": len(benches) > 0,
        "benchmark_count": len(benches),
    }


@app.get("/api/tri/{fund_id}")
def get_tri_fund(fund_id: int):
    perf = load_perf()
    meta = load_meta()
    benches = load_benchmarks()
    g = perf[perf["fund_id"] == fund_id]
    if g.empty:
        return JSONResponse(status_code=404, content={"error": f"fund {fund_id} not found"})
    name = dict(zip(meta["fund_id"], meta["fund_name"])).get(fund_id)
    s = fund_series(g)
    if s is None:
        return {"fund_id": fund_id, "fund_name": name, "dates": [], "series": [],
                "benchmark_available": len(benches) > 0}

    # Master axis = fund dates + every benchmark's months. The front end nulls
    # all series before the fund's inception and rebases to 100 there, so the
    # wide benchmark history doesn't widen the visible range.
    all_set = set(s["dates"])
    for bd in benches:
        all_set |= set(pd.Timestamp(d) for d in bd["df"]["month_end"])
    all_d = sorted(all_set)
    idx = {d: i for i, d in enumerate(all_d)}

    series = []
    fund_data = [None] * len(all_d)
    for d, v in zip(s["dates"], s["tri"]):
        fund_data[idx[d]] = _num(v)
    series.append({"fund_id": fund_id, "fund_name": name, "data": fund_data, "benchmark": False})

    for bd in benches:
        df = bd["df"]
        base = df["value"].iloc[0]
        data = [None] * len(all_d)
        if base and base != 0:
            for _, r in df.iterrows():
                data[idx[pd.Timestamp(r["month_end"])]] = _num(r["value"] / base * 100.0)
        series.append({"fund_id": None, "fund_name": bd["label"], "data": data,
                       "benchmark": True, "bench_key": bd["key"]})

    return {
        "fund_id": fund_id,
        "fund_name": name,
        "dates": [_datestr(d) for d in all_d],
        "series": series,
        "benchmark_available": len(benches) > 0,
    }


def _portfolio(perf, meta):
    """
    Purchase-amount-weighted portfolio calculation.

    For each month t, every SGG holding that has a non-null monthly_return that
    month and has been purchased by month t contributes. Its weight is its
    beginning-of-month NAV (prior-month sgg_nav, or its original PURCHASE_AMOUNT
    when there is no prior-month sgg_nav) divided by the sum of beginning-of-month
    NAVs across all contributing funds that month:

        portfolio_return_t = sum( w_i * monthly_return_i )

    Portfolio TRI chain-links these returns from base 100. Total SGG NAV is the
    sum of available sgg_nav each month. The allocation pie uses the most recent
    month where all eleven holdings report an sgg_nav. All series shown on the
    page are capped at that reference month. Nothing here hardcodes a date.
    """
    name = dict(zip(meta["fund_id"], meta["fund_name"]))
    holdings = list(PURCHASE_AMOUNTS.keys())  # the eleven SGG holdings

    # Per-fund raw maps: monthly_return and sgg_nav keyed by month_end.
    ret_map = {}  # fid -> {month: raw monthly_return}
    nav_map = {}  # fid -> {month: sgg_nav}
    first_sgg = {}  # fid -> first month with a non-null sgg_nav (purchase month)
    for fid, g in perf.groupby("fund_id"):
        fid = int(fid)
        g = g.sort_values("month_end")
        rm, nm = {}, {}
        for _, row in g.iterrows():
            if pd.isna(row["month_end"]):
                continue
            m = pd.Timestamp(row["month_end"])
            if pd.notna(row.get("monthly_return")):
                rm[m] = float(row["monthly_return"])
            if pd.notna(row.get("sgg_nav")) and float(row["sgg_nav"]) != 0:
                nm[m] = float(row["sgg_nav"])
        ret_map[fid] = rm
        nav_map[fid] = nm
        first_sgg[fid] = min(nm) if nm else None

    months_all = sorted({pd.Timestamp(d) for d in perf["month_end"].dropna().unique()})

    # Reference month for the pie + "data as of" = most recent month where ALL
    # eleven holdings report an sgg_nav.
    ref_month = None
    for t in reversed(months_all):
        if all(t in nav_map.get(fid, {}) for fid in holdings):
            ref_month = t
            break

    def le(d):
        return ref_month is None or d <= ref_month

    # Portfolio monthly returns, weighted by beginning-of-month NAV.
    port_dates, port_rets = [], []
    for k, t in enumerate(months_all):
        prev_t = months_all[k - 1] if k > 0 else None
        contribs = []  # (beginning_of_month_nav, monthly_return)
        for fid in holdings:
            # Fund must have a return this month and have been purchased by now.
            if t not in ret_map.get(fid, {}):
                continue
            if first_sgg[fid] is None or t < first_sgg[fid]:
                continue
            prev_nav = nav_map[fid].get(prev_t) if prev_t is not None else None
            bom = prev_nav if prev_nav is not None else PURCHASE_AMOUNTS[fid]
            contribs.append((bom, ret_map[fid][t]))
        if not contribs:
            continue
        denom = sum(b for b, _ in contribs)
        if denom <= 0:
            continue
        r = sum((b / denom) * ret for b, ret in contribs)
        port_dates.append(t)
        port_rets.append(r)

    # Cap the return series at the reference month (no partial future months).
    capped = [(d, r) for d, r in zip(port_dates, port_rets) if le(d)]
    port_dates = [d for d, _ in capped]
    port_rets = [r for _, r in capped]

    # Portfolio TRI: base 100 at first month with a return, chain-link forward.
    ptri_dates, ptri = [], []
    if port_dates:
        ptri_dates.append(port_dates[0])
        ptri.append(100.0)
        for i in range(1, len(port_dates)):
            ptri.append(ptri[-1] * (1.0 + port_rets[i]))
            ptri_dates.append(port_dates[i])

    # Total SGG NAV = sum of every holding's sgg_nav available that month,
    # capped at the reference month.
    tot_dates, tot_vals = [], []
    for m in months_all:
        if not le(m):
            continue
        vals = [nav_map[fid][m] for fid in holdings if m in nav_map.get(fid, {})]
        if vals:
            tot_dates.append(m)
            tot_vals.append(sum(vals))

    # Allocation pie: each holding's sgg_nav in the reference month.
    alloc = []
    if ref_month is not None:
        for fid in holdings:
            v = nav_map.get(fid, {}).get(ref_month)
            if v is not None:
                alloc.append({"fund_id": fid, "fund_name": name.get(fid), "sgg_nav": v})
        tot_alloc = sum(a["sgg_nav"] for a in alloc) or 1.0
        for a in alloc:
            a["pct"] = _num(a["sgg_nav"] / tot_alloc * 100.0)
        alloc.sort(key=lambda x: -x["sgg_nav"])

    return {
        "port_dates": port_dates,
        "port_rets": port_rets,
        "ptri_dates": ptri_dates,
        "ptri": ptri,
        "tot_dates": tot_dates,
        "tot_vals": tot_vals,
        "alloc": alloc,
        "nav_map": nav_map,
        "ref_month": ref_month,
    }


@app.get("/api/portfolio")
def get_portfolio():
    perf = load_perf()
    meta = load_meta()
    p = _portfolio(perf, meta)
    ref = p["ref_month"]
    return {
        # "Portfolio data as of" reference month (most recent month with all
        # eleven holdings reporting). Everything on the page is capped to it.
        "as_of": _datestr(ref) if ref is not None else None,
        "monthly": {
            "dates": [_datestr(d) for d in p["port_dates"]],
            "returns": [_num(r) for r in p["port_rets"]],
        },
        "tri": {
            "dates": [_datestr(d) for d in p["ptri_dates"]],
            "values": [_num(v) for v in p["ptri"]],
        },
        "total_nav": {
            "dates": [_datestr(d) for d in p["tot_dates"]],
            "values": [_num(v) for v in p["tot_vals"]],
        },
        "allocation": [
            {"fund_id": a["fund_id"], "fund_name": a["fund_name"], "sgg_nav": _num(a["sgg_nav"]), "pct": a["pct"]}
            for a in p["alloc"]
        ],
    }


@app.get("/api/portfolio_tri")
def get_portfolio_tri():
    """Portfolio TRI (base 100) plus each benchmark's TRI (base 100) on a shared
    monthly axis. The front end rebases everything to 100 at the portfolio's first
    month and toggles series, exactly like the home Normalized Total Returns chart."""
    perf = load_perf()
    meta = load_meta()
    p = _portfolio(perf, meta)
    benches = load_benchmarks()

    pdates = p["ptri_dates"]
    pvals = p["ptri"]
    all_set = set(pd.Timestamp(d) for d in pdates)
    for bd in benches:
        all_set |= set(pd.Timestamp(d) for d in bd["df"]["month_end"])
    all_d = sorted(all_set)
    idx = {d: i for i, d in enumerate(all_d)}

    series = []
    pdata = [None] * len(all_d)
    for d, v in zip(pdates, pvals):
        pdata[idx[pd.Timestamp(d)]] = _num(v)
    series.append({"fund_id": None, "fund_name": "SGG Portfolio", "data": pdata, "benchmark": False})

    for bd in benches:
        df = bd["df"]
        base = df["value"].iloc[0]
        data = [None] * len(all_d)
        if base and base != 0:
            for _, r in df.iterrows():
                data[idx[pd.Timestamp(r["month_end"])]] = _num(r["value"] / base * 100.0)
        series.append({"fund_id": None, "fund_name": bd["label"], "data": data,
                       "benchmark": True, "bench_key": bd["key"]})

    return {"dates": [_datestr(d) for d in all_d], "series": series}


@app.get("/api/risk_metrics")
def get_risk_metrics(common: bool = False):
    perf = load_perf()
    meta = load_meta()
    bench = load_benchmark()
    name = dict(zip(meta["fund_id"], meta["fund_name"]))
    bret = benchmark_returns(bench)

    # When common=True (Risk/Return comparison tables) every fund is truncated to
    # the last month all funds report, so all metrics share one as-of endpoint.
    cutoff = _all_funds_cutoff(perf) if common else None

    rows = []
    for fid, g in perf.groupby("fund_id"):
        fid = int(fid)
        if cutoff is not None:
            g = g[g["month_end"] <= cutoff]
        s = fund_series(g)
        if s is None:
            continue
        dd = max_drawdown(s["dates"], s["tri"])

        rets = [r for r in s["returns"] if r is not None]
        std = _num(np.std(rets, ddof=1) * math.sqrt(12)) if len(rets) >= 2 else None

        # Downside capture vs benchmark in benchmark-down months.
        #
        # Methodology (must stay dynamic — reads all rows, no hardcoded dates):
        #   1. Align the fund's RAW monthly_return series with the benchmark's
        #      calculable monthly returns, by month_end. Only months where BOTH
        #      the fund has a non-null monthly_return AND the benchmark has a
        #      calculable return are included ("aligned history").
        #   2. Benchmark down month = aligned month where benchmark return < 0.
        #   3. If there are zero benchmark down months -> "n/a".
        #      Else downside_capture_pct =
        #         mean(fund returns in down months)
        #         / mean(benchmark returns in down months) * 100
        #   4. Return the pct rounded to 1 dp as a float, or the string "n/a".
        #      Never null.
        dcr = downside_capture(g, bret)

        rows.append(
            {
                "fund_id": fid,
                "fund_name": name.get(fid),
                "max_drawdown": dd["max_drawdown"],
                "peak_date": dd["peak_date"],
                "trough_date": dd["trough_date"],
                "time_to_trough": dd["time_to_trough"],
                "recovery_date": dd["recovery_date"],
                "recovery_time": dd["recovery_time"],
                "annualized_std_dev": std,
                "downside_capture_ratio": dcr,
                "as_of": _datestr(s["dates"][-1]),
            }
        )
    rows.sort(key=lambda x: (x["fund_name"] or "").lower())
    return {
        "count": len(rows),
        "benchmark_available": bench is not None,
        "as_of": _datestr(cutoff) if cutoff is not None else None,
        "metrics": rows,
        "index_metrics": _index_metrics(cutoff, bret),
    }


@app.get("/api/return_metrics")
def get_return_metrics(common: bool = False):
    perf = load_perf()
    meta = load_meta()
    name = dict(zip(meta["fund_id"], meta["fund_name"]))

    # When common=True (Risk/Return comparison tables) every fund is truncated to
    # the last month all funds report, so all metrics share one as-of endpoint.
    # The home-page "Last Monthly Return by Fund" bars call WITHOUT common, so the
    # common_month_return / last_return_as_of logic below is unaffected for them.
    cutoff = _all_funds_cutoff(perf) if common else None

    # Common as-of month for the "Last Monthly Return by Fund" comparison.
    # = the most recent month_end that has at least one reported monthly_return
    # across all funds. Dynamic: auto-advances as funds post new returns, so the
    # bars always share a single, freshest available as-of date rather than each
    # fund showing its own latest (which mixes months).
    reported = perf[perf["monthly_return"].notna()]
    common_as_of = pd.Timestamp(reported["month_end"].max()) if not reported.empty else None

    rows = []
    for fid, g in perf.groupby("fund_id"):
        fid = int(fid)
        if cutoff is not None:
            g = g[g["month_end"] <= cutoff]
        s = fund_series(g)
        if s is None:
            continue
        dates = s["dates"]
        tri = [v for v in s["tri"] if v is not None]
        inception = s["inception"]
        latest = dates[-1]
        msi = _months_between(inception, latest)

        tri_final = tri[-1] if tri else None
        total_ret = _num(tri_final / 100.0 - 1.0) if tri_final else None
        cagr = _num((tri_final / 100.0) ** (12.0 / msi) - 1.0) if (tri_final and msi > 0) else None

        # Effective monthly returns with their dates (drop None base months).
        rd = [(pd.Timestamp(d), r) for d, r in zip(s["dates"], s["returns"]) if r is not None]
        rets = [r for _, r in rd]
        trailing_1y = _num(_product(rets[-12:])) if len(rets) >= 12 else None
        trailing_3y = _num(_product(rets[-36:])) if len(rets) >= 36 else None

        latest_year = latest.year
        ytd_rets = [r for d, r in rd if d.year == latest_year]
        ytd = _num(_product(ytd_rets)) if ytd_rets else None

        last_mr = rets[-1] if rets else None

        # Return for the shared common as-of month (None if this fund hasn't
        # reported that month yet -> rendered as greyed/pending in the chart).
        common_mr = None
        if common_as_of is not None:
            for d, r in rd:
                if d == common_as_of:
                    common_mr = r
                    break

        rows.append(
            {
                "fund_id": fid,
                "fund_name": name.get(fid),
                "ytd_return": ytd,
                "trailing_1y": trailing_1y,
                "trailing_3y": trailing_3y,
                "annualized_since_inception": cagr,
                "total_return_since_inception": total_ret,
                "months_since_inception": msi,
                "last_monthly_return": _num(last_mr),
                "common_month_return": _num(common_mr),
                "as_of": _datestr(latest),
            }
        )
    # Benchmark index monthly returns for the common as-of month (used by the
    # home "Last Monthly Return by Fund" chart). Null when an index has no data
    # for that month (e.g. the Pitchbook/Morningstar index publishes on a lag).
    bench_returns = []
    if common_as_of is not None:
        prev_m = pd.Timestamp(common_as_of) - pd.offsets.MonthEnd(1)
        for bd in load_benchmarks():
            m = {pd.Timestamp(r["month_end"]): float(r["value"]) for _, r in bd["df"].iterrows()}
            cur = m.get(pd.Timestamp(common_as_of))
            pv = m.get(prev_m)
            mr = _num(cur / pv - 1.0) if (cur is not None and pv) else None
            bench_returns.append({"key": bd["key"], "label": bd["label"], "monthly_return": mr})

    rows.sort(key=lambda x: (x["fund_name"] or "").lower())
    bret = benchmark_returns(load_benchmark())
    return {
        "count": len(rows),
        "as_of": _datestr(cutoff) if cutoff is not None else None,
        "last_return_as_of": _datestr(common_as_of) if common_as_of is not None else None,
        "benchmark_returns": bench_returns,
        "metrics": rows,
        "index_metrics": _index_metrics(cutoff, bret),
    }


@app.get("/api/kpis")
def get_kpis():
    perf = load_perf()
    meta = load_meta()
    p = _portfolio(perf, meta)

    holdings = sum(1 for fid in p["nav_map"] if p["nav_map"][fid])
    last_port = _num(p["port_rets"][-1]) if p["port_rets"] else None
    current_total = _num(p["tot_vals"][-1]) if p["tot_vals"] else None
    last_port_as_of = _datestr(p["port_dates"][-1]) if p["port_dates"] else None
    current_total_as_of = _datestr(p["tot_dates"][-1]) if p["tot_dates"] else None

    # YTD portfolio return = compounded portfolio monthly returns in the latest
    # portfolio month's calendar year, as of that month.
    ytd_port = None
    ytd_port_as_of = None
    if p["port_dates"]:
        last_d = pd.Timestamp(p["port_dates"][-1])
        yr_rets = [r for d, r in zip(p["port_dates"], p["port_rets"])
                   if pd.Timestamp(d).year == last_d.year]
        if yr_rets:
            ytd_port = _num(_product(yr_rets))
            ytd_port_as_of = _datestr(last_d)

    # Last-available-month return of the Pitchbook benchmark (most recent
    # month-over-month change in the pitchbook index), with its as-of month.
    bench = load_benchmark()
    bench_ret = None
    bench_ret_as_of = None
    if bench is not None and len(bench) >= 2:
        b = bench.sort_values("month_end").reset_index(drop=True)
        prev = b["pitchbook_index"].iloc[-2]
        cur = b["pitchbook_index"].iloc[-1]
        if prev and prev != 0:
            bench_ret = _num(cur / prev - 1.0)
            bench_ret_as_of = _datestr(b["month_end"].iloc[-1])

    return {
        "holdings_tracked": holdings,
        "last_month_portfolio_return": last_port,
        "last_month_portfolio_return_as_of": last_port_as_of,
        "current_total_sgg_nav": current_total,
        "current_total_sgg_nav_as_of": current_total_as_of,
        "last_month_benchmark_return": bench_ret,
        "last_month_benchmark_return_as_of": bench_ret_as_of,
        "ytd_portfolio_return": ytd_port,
        "ytd_portfolio_return_as_of": ytd_port_as_of,
    }


@app.get("/api/logos")
def get_logos():
    """Map fund_id (as string) -> /static/images/<file>, or null if no file."""
    out = {}
    for fid, base in LOGO_BASENAMES.items():
        path = None
        for ext in SUPPORTED_EXTENSIONS:
            if os.path.exists(f"static/images/{base}{ext}"):
                path = f"/static/images/{base}{ext}"
                break
        out[str(fid)] = path
    return out


# ───────────────────────────────────────────────────────────── frontend ──
@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
# (v11 benchmark wiring: pitchbook + stepstone_pm + sp500)
