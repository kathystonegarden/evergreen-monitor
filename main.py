"""
SGG Evergreen Monitor — minimal FastAPI backend.

Proves the Excel-to-website pipeline: the Evergreen workbook is read fresh on
every request (no caching) so spreadsheet edits show up immediately. Serves a
single static dashboard page.
"""

import os

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Source workbook. Override with EXCEL_PATH in deployment.
EXCEL_PATH = os.environ.get(
    "EXCEL_PATH", "data/06_05_26_Evergreen_Database_v7.xlsx"
)

PERFORMANCE_SHEET = "fund_performance"
METADATA_SHEET = "fund_metadata"

app = FastAPI(title="SGG Evergreen Monitor")


def _read_sheet(sheet_name: str) -> pd.DataFrame:
    """Read one sheet fresh from disk (no caching)."""
    return pd.read_excel(EXCEL_PATH, sheet_name=sheet_name, engine="openpyxl")


@app.get("/api/funds")
def get_funds():
    """Return the fund list from the metadata sheet."""
    try:
        meta = _read_sheet(METADATA_SHEET)
    except FileNotFoundError:
        return JSONResponse(
            status_code=500,
            content={"error": f"Excel file not found at {EXCEL_PATH}"},
        )

    cols = [c for c in ["fund_id", "fund_name", "strategy", "ticker"] if c in meta.columns]
    funds = meta[cols].where(pd.notnull(meta[cols]), None).to_dict(orient="records")
    return {"count": len(funds), "funds": funds}


@app.get("/api/nav")
def get_nav():
    """
    Return the latest NAV/share per fund: for each fund_id, the most recent
    month_end where nav_per_share is not null, joined to fund_name + strategy
    from the metadata sheet on fund_id.
    """
    try:
        perf = _read_sheet(PERFORMANCE_SHEET)
        meta = _read_sheet(METADATA_SHEET)
    except FileNotFoundError:
        return JSONResponse(
            status_code=500,
            content={"error": f"Excel file not found at {EXCEL_PATH}"},
        )

    # Keep only rows that have a NAV/share value and a valid date.
    perf = perf[["fund_id", "month_end", "nav_per_share"]].copy()
    perf = perf[perf["nav_per_share"].notnull()]
    perf["month_end"] = pd.to_datetime(perf["month_end"], errors="coerce")
    perf = perf[perf["month_end"].notnull()]

    # Most recent month per fund.
    latest = (
        perf.sort_values("month_end")
        .groupby("fund_id", as_index=False)
        .last()
    )

    # Join fund names + strategy from metadata on fund_id.
    meta_cols = [c for c in ["fund_id", "fund_name", "strategy"] if c in meta.columns]
    merged = latest.merge(meta[meta_cols], on="fund_id", how="left")

    rows = [
        {
            "fund_id": int(r["fund_id"]),
            "fund_name": r["fund_name"] if pd.notnull(r.get("fund_name")) else None,
            "strategy": r["strategy"] if pd.notnull(r.get("strategy")) else None,
            "as_of": r["month_end"].strftime("%Y-%m-%d"),
            "nav_per_share": round(float(r["nav_per_share"]), 4),
        }
        for _, r in merged.iterrows()
    ]
    rows.sort(key=lambda x: (x["fund_name"] or "").lower())
    return {"count": len(rows), "navs": rows}


@app.get("/")
def index():
    """Serve the dashboard."""
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
