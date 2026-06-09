"""
Standalone verification of the portfolio monthly-return methodology in main.py.

Run from the repo root:

    python verify_portfolio.py

Loads the live Excel database, runs main._portfolio(), and checks portfolio
monthly returns, Total SGG NAV, and the allocation pie against the expected
values from the methodology file. Exits 0 if everything matches, 1 otherwise.
"""

import main

EXPECTED_RETURNS = {  # month -> percent (1 + return) * 100 displayed as %
    "2025-12": 0.82,
    "2026-01": 0.69,
    "2026-02": 1.09,
    "2026-03": 0.31,
}
EXPECTED_TOTAL_NAV = {
    "2025-12": 24554555.61,
    "2026-01": 85065685.25,
    "2026-02": 91112576.15,
    "2026-03": 86147422.08,
}
EXPECTED_PIE = {  # fund_id -> sgg_nav in reference month
    1: 5114594.58,
    2: 5078764.96,
    3: 4564503.76,
    4: 10396928.17,
    5: 4975704.57,
    6: 9974021.75,
    7: 5505418.37,
    8: 9989376.78,
    9: 10242797.01,
    10: 10211436.28,
    11: 10093875.85,
}
PIE_TOTAL = 86147422.08


def ym(ts):
    return f"{ts.year:04d}-{ts.month:02d}"


def approx(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol


def run():
    perf = main.load_perf()
    meta = main.load_meta()
    p = main._portfolio(perf, meta)
    ok = True

    print("=== Portfolio monthly returns (%) ===")
    rets = {ym(d): r * 100 for d, r in zip(p["port_dates"], p["port_rets"])}
    for m, exp in EXPECTED_RETURNS.items():
        got = rets.get(m)
        match = approx(got, exp, 0.005)  # within half a basis point of the 2dp value
        ok = ok and match
        print(f"  {m}: got {got:.2f}%  expected {exp:.2f}%  {'OK' if match else 'MISMATCH'}"
              if got is not None else f"  {m}: MISSING  expected {exp:.2f}%  MISMATCH")

    print("=== Total SGG NAV ($) ===")
    nav = {ym(d): v for d, v in zip(p["tot_dates"], p["tot_vals"])}
    for m, exp in EXPECTED_TOTAL_NAV.items():
        got = nav.get(m)
        match = approx(got, exp, 0.01)
        ok = ok and match
        print(f"  {m}: got {got:,.2f}  expected {exp:,.2f}  {'OK' if match else 'MISMATCH'}"
              if got is not None else f"  {m}: MISSING  expected {exp:,.2f}  MISMATCH")

    print(f"=== Allocation pie (reference month {main._datestr(p['ref_month'])}) ===")
    pie = {a["fund_id"]: a["sgg_nav"] for a in p["alloc"]}
    for fid, exp in EXPECTED_PIE.items():
        got = pie.get(fid)
        match = approx(got, exp, 0.01)
        ok = ok and match
        print(f"  fund {fid:>2}: got {got:,.2f}  expected {exp:,.2f}  {'OK' if match else 'MISMATCH'}"
              if got is not None else f"  fund {fid:>2}: MISSING  expected {exp:,.2f}  MISMATCH")
    pie_total = sum(pie.values())
    tmatch = approx(pie_total, PIE_TOTAL, 0.02)
    ok = ok and tmatch
    print(f"  TOTAL : got {pie_total:,.2f}  expected {PIE_TOTAL:,.2f}  {'OK' if tmatch else 'MISMATCH'}")

    print("\nALL MATCH ✓" if ok else "\nMISMATCHES FOUND ✗")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
