"""Fetch OHLCV data from Yahoo Finance in pull.js CSV format.

CI-safe drop-in for local pull.js (which needs a running TradingView app).
Symbols: EURUSD=X, XAUUSD=GC=F (COMEX gold futures as a spot proxy).

Usage:
    python fetch_data.py --symbol EURUSD=X --tf D --out data/eurusd_daily.csv
    python fetch_data.py --symbol GC=F --tf 240 --out data/xauusd_4h.csv
"""
import argparse
import json
import time
import urllib.request

import numpy as np

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

RANGES = {"D": "10y", "240": "730d", "60": "2y", "1h": "2y"}


def fetch(symbol, interval, rng):
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval={interval}&range={rng}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    result = data["chart"]["result"][0]
    ts = result.get("timestamp") or []
    quote = result["indicators"]["quote"][0]
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    vols = quote.get("volume") or []
    rows = []
    for i, t in enumerate(ts):
        if i >= len(closes) or closes[i] is None:
            continue
        rows.append(
            (
                int(t),
                float(opens[i]),
                float(highs[i]),
                float(lows[i]),
                float(closes[i]),
                int(vols[i]) if vols[i] else 0,
            )
        )
    arr = np.array(rows, dtype=np.float64)
    return arr


def resample_4h(arr):
    """Resample 1h bars to 4h aligned on UTC 00:00/04:00/08:00..., closing gaps."""
    buckets = {}
    for t, o, h, l, c, v in arr:
        b = (int(t) // 14400) * 14400
        if b not in buckets:
            buckets[b] = [o, h, l, c, v]
        else:
            buckets[b][1] = max(buckets[b][1], h)
            buckets[b][2] = min(buckets[b][2], l)
            buckets[b][3] = c  # last close
            buckets[b][4] += v
    out = np.array([[b] + buckets[b] for b in sorted(buckets)], dtype=np.float64)
    return out


def write_csv(path, arr):
    with open(path, "w") as fh:
        fh.write("time,open,high,low,close,volume\n")
        for t, o, h, l, c, v in arr:
            fh.write(f"{int(t)},{o:.5f},{h:.5f},{l:.5f},{c:.5f},{int(v)}\n")


def main_driver(symbol, tf, out):
    """Fetch fresh bars for symbol and write OHLCV CSV at path (importable helper)."""
    if tf == "240":
        interval = "60m"
    elif tf == "D":
        interval = "1d"
    else:
        raise SystemExit("only --tf D or 240 supported")
    arr = fetch(symbol, interval, RANGES[tf])
    if tf == "240":
        arr = resample_4h(arr)
    write_csv(out, arr)
    print(f"Wrote {len(arr)} bars to {out}")
    return arr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--tf", required=True, help="D or 240")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main_driver(args.symbol, args.tf, args.out)


if __name__ == "__main__":
    main()