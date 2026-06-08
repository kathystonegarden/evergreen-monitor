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

EXCEL_PATH = os.environ.get("EXCEL_PATH", "data/06_07_26_Evergreen_Database_v8.xlsx")
PERF_SHEET = "fund_performance"
META_SHEET = "fund_metadata"
PB_SHEET = "pitchbook_tri_index"

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

    Returns dict: dates[], returns[] (effective monthly returns, None at base),
    tri[] (base 100 at inception), inception(Timestamp), or None if no data.
    """
    g = g.sort_values("month_end").reset_index(drop=True)
    nav = g["nav_per_share"]
    ret = g["monthly_return"]
    dist = g["distributions_per_share"].fillna(0.0)

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
            if pd.notna(r):
                r = float(r)
            elif (
                pd.notna(nav.iloc[i])
                and pd.notna(nav.iloc[i - 1])
                and nav.iloc[i - 1] != 0
            ):
                r = float((nav.iloc[i] + dist.iloc[i] - nav.iloc[i - 1]) / nav.iloc[i - 1])
            else:
                r = None
            if r is not None:
                cur = cur * (1.0 + r)
        dates.append(g["month_end"].iloc[i])
        returns.append(r)
        tri.append(cur)

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


# ─────────────────────────────────────────────────────────────── endpoints ──
@app.get("/api/funds")
def get_funds():
    meta = load_meta()
    cols = ["fund_id", "fund_name", "strategy", "ticker", "class_inception_date"]
    cols = [c for c in cols if c in meta.columns]
    rows = []
    for _, r in meta[cols].iterrows():
        rows.append(
            {
                "fund_id": int(r["fund_id"]),
                "fund_name": r.get("fund_name") if pd.notna(r.get("fund_name")) else None,
                "strategy": r.get("strategy") if pd.notna(r.get("strategy")) else None,
                "ticker": r.get("ticker") if pd.notna(r.get("ticker")) else None,
                "class_inception_date": _datestr(r["class_inception_date"])
                if pd.notna(r.get("class_inception_date"))
                else None,
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
    bench = load_benchmark()
    series = _all_fund_series(perf, meta)

    # Master date axis = union of all fund dates (+ benchmark).
    all_dates = set()
    for s in series.values():
        all_dates.update(s["dates"])
    if bench is not None:
        all_dates.update(pd.Timestamp(d) for d in bench["month_end"])
    dates = sorted(all_dates)
    idx = {d: i for i, d in enumerate(dates)}

    out_series = []
    for fid, s in series.items():
        data = [None] * len(dates)
        for d, v in zip(s["dates"], s["tri"]):
            data[idx[d]] = _num(v)
        out_series.append({"fund_id": fid, "fund_name": s["name"], "data": data, "benchmark": False})

    if bench is not None and len(bench) >= 1:
        base = bench["pitchbook_index"].iloc[0]
        data = [None] * len(dates)
        for _, r in bench.iterrows():
            data[idx[pd.Timestamp(r["month_end"])]] = _num(r["pitchbook_index"] / base * 100.0)
        out_series.append(
            {"fund_id": None, "fund_name": "Pitchbook Index", "data": data, "benchmark": True}
        )

    return {
        "dates": [_datestr(d) for d in dates],
        "series": out_series,
        "benchmark_available": bench is not None,
    }


@app.get("/api/tri/{fund_id}")
def get_tri_fund(fund_id: int):
    perf = load_perf()
    meta = load_meta()
    bench = load_benchmark()
    g = perf[perf["fund_id"] == fund_id]
    if g.empty:
        return JSONResponse(status_code=404, content={"error": f"fund {fund_id} not found"})
    name = dict(zip(meta["fund_id"], meta["fund_name"])).get(fund_id)
    s = fund_series(g)
    if s is None:
        return {"fund_id": fund_id, "fund_name": name, "dates": [], "fund": None, "benchmark": None}

    dates = list(s["dates"])
    if bench is not None:
        all_d = sorted(set(dates) | set(pd.Timestamp(d) for d in bench["month_end"]))
    else:
        all_d = dates
    idx = {d: i for i, d in enumerate(all_d)}

    fund_data = [None] * len(all_d)
    for d, v in zip(s["dates"], s["tri"]):
        fund_data[idx[d]] = _num(v)

    bench_data = None
    if bench is not None and len(bench) >= 1:
        base = bench["pitchbook_index"].iloc[0]
        bench_data = [None] * len(all_d)
        for _, r in bench.iterrows():
            bench_data[idx[pd.Timestamp(r["month_end"])]] = _num(r["pitchbook_index"] / base * 100.0)

    return {
        "fund_id": fund_id,
        "fund_name": name,
        "dates": [_datestr(d) for d in all_d],
        "fund": {"fund_name": name, "data": fund_data},
        "benchmark": ({"fund_name": "Pitchbook Index", "data": bench_data} if bench_data else None),
        "benchmark_available": bench is not None,
    }


def _portfolio(perf, meta):
    name = dict(zip(meta["fund_id"], meta["fund_name"]))
    series = _all_fund_series(perf, meta)

    # Per fund: effective monthly returns + raw sgg_nav by month.
    ret_map = {}  # fid -> {month: return}
    for fid, s in series.items():
        ret_map[fid] = {
            pd.Timestamp(d): r for d, r in zip(s["dates"], s["returns"]) if r is not None
        }
    nav_map = {}  # fid -> {month: sgg_nav}
    for fid, g in perf.groupby("fund_id"):
        fid = int(fid)
        m = {}
        for _, row in g.iterrows():
            if pd.notna(row["sgg_nav"]) and float(row["sgg_nav"]) > 0:
                m[pd.Timestamp(row["month_end"])] = float(row["sgg_nav"])
        nav_map[fid] = m

    months_all = sorted({pd.Timestamp(d) for d in perf["month_end"].unique()})

    # NAV-weighted portfolio monthly returns (>=3 contributors required).
    port_dates, port_rets = [], []
    for k in range(1, len(months_all)):
        t, tp = months_all[k], months_all[k - 1]
        contribs = [
            fid
            for fid in series
            if tp in nav_map.get(fid, {}) and t in ret_map.get(fid, {})
        ]
        if len(contribs) >= 3:
            total = sum(nav_map[fid][tp] for fid in contribs)
            if total > 0:
                r = sum((nav_map[fid][tp] / total) * ret_map[fid][t] for fid in contribs)
                port_dates.append(t)
                port_rets.append(r)

    # Portfolio TRI: base 100 at first calculable month, chain-link forward.
    ptri_dates, ptri = [], []
    if port_dates:
        ptri_dates.append(port_dates[0])
        ptri.append(100.0)
        for i in range(1, len(port_dates)):
            ptri.append(ptri[-1] * (1.0 + port_rets[i]))
            ptri_dates.append(port_dates[i])

    # Total SGG NAV by month.
    tot_dates, tot_vals = [], []
    for m in months_all:
        s = sum(nav_map[fid].get(m, 0.0) for fid in nav_map)
        if s > 0:
            tot_dates.append(m)
            tot_vals.append(s)

    # Current allocation = latest sgg_nav per fund.
    alloc = []
    for fid in nav_map:
        if nav_map[fid]:
            last_m = max(nav_map[fid])
            alloc.append({"fund_id": fid, "fund_name": name.get(fid), "sgg_nav": nav_map[fid][last_m]})
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
    }


@app.get("/api/portfolio")
def get_portfolio():
    perf = load_perf()
    meta = load_meta()
    p = _portfolio(perf, meta)
    return {
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


@app.get("/api/risk_metrics")
def get_risk_metrics():
    perf = load_perf()
    meta = load_meta()
    bench = load_benchmark()
    name = dict(zip(meta["fund_id"], meta["fund_name"]))
    bret = benchmark_returns(bench)

    rows = []
    for fid, g in perf.groupby("fund_id"):
        fid = int(fid)
        s = fund_series(g)
        if s is None:
            continue
        dd = max_drawdown(s["dates"], s["tri"])

        rets = [r for r in s["returns"] if r is not None]
        std = _num(np.std(rets, ddof=1) * math.sqrt(12)) if len(rets) >= 2 else None

        # Downside capture vs benchmark in benchmark-down months.
        dcr = None
        if bret:
            fr = {pd.Timestamp(d): r for d, r in zip(s["dates"], s["returns"]) if r is not None}
            down = [(fr[m], bret[m]) for m in fr if m in bret and bret[m] < 0]
            if down:
                f_avg = np.mean([d[0] for d in down])
                b_avg = np.mean([d[1] for d in down])
                if b_avg != 0:
                    dcr = _num(f_avg / b_avg * 100.0)

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
            }
        )
    rows.sort(key=lambda x: (x["fund_name"] or "").lower())
    return {"count": len(rows), "benchmark_available": bench is not None, "metrics": rows}


@app.get("/api/return_metrics")
def get_return_metrics():
    perf = load_perf()
    meta = load_meta()
    name = dict(zip(meta["fund_id"], meta["fund_name"]))

    rows = []
    for fid, g in perf.groupby("fund_id"):
        fid = int(fid)
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
            }
        )
    rows.sort(key=lambda x: (x["fund_name"] or "").lower())
    return {"count": len(rows), "metrics": rows}


@app.get("/api/kpis")
def get_kpis():
    perf = load_perf()
    meta = load_meta()
    p = _portfolio(perf, meta)

    holdings = sum(1 for fid in p["nav_map"] if p["nav_map"][fid])
    last_port = _num(p["port_rets"][-1]) if p["port_rets"] else None
    current_total = _num(p["tot_vals"][-1]) if p["tot_vals"] else None

    return {
        "holdings_tracked": holdings,
        "last_month_portfolio_return": last_port,
        "current_total_sgg_nav": current_total,
    }


# ───────────────────────────────────────────────────────────── frontend ──
@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
