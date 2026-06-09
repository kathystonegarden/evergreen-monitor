"""
Standalone verification of the downside-capture methodology in main.py.

Run from the repo root:

    python verify_downside_capture.py

It loads the live Excel database (via main.EXCEL_PATH), computes downside
capture for all funds using main.downside_capture(), and compares the results
against the expected values from the methodology file. Exits 0 if every fund
matches, 1 otherwise.
"""

import main

# Expected results from the methodology file.
EXPECTED = {
    1: -198.8,
    2: 83.8,
    3: -191.2,
    4: "n/a",
    5: "n/a",
    6: -53.1,
    7: -9.3,
    8: 49.6,
    9: "n/a",
    10: 12.3,
    11: 178.2,
}


def run():
    perf = main.load_perf()
    bench = main.load_benchmark()
    bret = main.benchmark_returns(bench)

    all_ok = True
    print(f"{'fund':>4}  {'result':>9}  {'expected':>9}  match")
    print("-" * 38)
    for fid, g in perf.groupby("fund_id"):
        fid = int(fid)
        result = main.downside_capture(g, bret)
        expected = EXPECTED.get(fid)
        match = result == expected
        all_ok = all_ok and match
        print(f"{fid:>4}  {str(result):>9}  {str(expected):>9}  {'OK' if match else 'MISMATCH'}")

    print("-" * 38)
    print("ALL MATCH ✓" if all_ok else "MISMATCHES FOUND ✗")
    return all_ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
