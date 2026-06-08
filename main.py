from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import os

app = FastAPI()

# ── load data ──────────────────────────────────────────────────────────────────
EXCEL_PATH = os.getenv("EXCEL_PATH", "data/06_05_26_Evergreen_Database_v7.xlsx")

def load_data():
    perf = pd.read_excel(EXCEL_PATH, sheet_name="fund_performance", engine="openpyxl")
    meta = pd.read_excel(EXCEL_PATH, sheet_name="fund_metadata",    engine="openpyxl")

    # drop formula-only rows (fund_name comes from VLOOKUP in Excel, merge from meta instead)
    perf = perf.drop(columns=["fund_name", "strategy"], errors="ignore")
    df = perf.merge(meta[["fund_id", "fund_name", "strategy"]], on="fund_id", how="left")

    df["month_end"] = pd.to_datetime(df["month_end"]).dt.strftime("%Y-%m-%d")
    return df, meta

# ── API routes ─────────────────────────────────────────────────────────────────
@app.get("/api/nav")
def get_nav():
    """Latest NAV/share for every fund — powers the summary table."""
    df, _ = load_data()
    latest = (
        df[df["nav_per_share"].notna()]
        .sort_values("month_end", ascending=False)
        .groupby("fund_id")
        .first()
        .reset_index()
    )
    return latest[["fund_id", "fund_name", "strategy", "month_end", "nav_per_share", "sgg_nav"]].to_dict(orient="records")


@app.get("/api/nav_history/{fund_id}")
def get_nav_history(fund_id: int):
    """Full NAV/share time series for a single fund — powers the line chart."""
    df, _ = load_data()
    fund = df[(df["fund_id"] == fund_id) & df["nav_per_share"].notna()]
    fund = fund.sort_values("month_end")
    return fund[["month_end", "nav_per_share", "fund_name"]].to_dict(orient="records")


@app.get("/api/funds")
def get_funds():
    """Fund list for the dropdown."""
    _, meta = load_data()
    return meta[["fund_id", "fund_name", "strategy", "ticker"]].to_dict(orient="records")


# ── serve frontend ─────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")
