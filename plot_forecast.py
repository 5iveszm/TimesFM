"""Plot a TimesFM-3 forecast against recent history.

Usage:
    python plot_forecast.py --csv data/eurusd_daily.csv \
        --forecast data/eurusd_daily_forecast_14.csv \
        --out plot_eurusd_daily.png --history 120
"""
import argparse
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")  # headless-safe for CI
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def to_datenum(times):
    return mdates.date2num(
        np.array([datetime.fromtimestamp(int(t), tz=timezone.utc) for t in times])
    )


def load_csv(path):
    with open(path) as fh:
        header = next(fh).strip().split(",")
        rows = [line.strip().split(",") for line in fh if line.strip()]
    times = np.array([int(r[0].split(".")[0]) for r in rows], dtype=np.int64)
    close = np.array([float(r[4]) for r in rows])
    return header, times, close


def load_forecast(path):
    with open(path) as fh:
        header = next(fh).strip().split(",")
        rows = [line.strip().split(",") for line in fh if line.strip()]
    cols = {name: i for i, name in enumerate(header)}
    med = np.array([float(r[cols["close_median"]]) for r in rows])
    q10 = np.array([float(r[cols["close_q10"]]) for r in rows]) if "close_q10" in cols else None
    q90 = np.array([float(r[cols["close_q90"]]) for r in rows]) if "close_q90" in cols else None
    return med, q10, q90


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="historical OHLCV CSV from pull.js")
    ap.add_argument("--forecast", required=True, help="forecast CSV from forecast.py")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--history", type=int, default=120, help="history bars to show")
    args = ap.parse_args()

    _, times, close = load_csv(args.csv)
    med, q10, q90 = load_forecast(args.forecast)

    hist = close[-args.history :]
    hist_t = times[-args.history :]

    # Build forecast time axis using the average bar spacing of recent history.
    dt = int(np.mean(np.diff(hist_t)))
    base = hist_t[-1]
    fut_t = np.array([base + dt * (i + 1) for i in range(len(med))])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        to_datenum(hist_t),
        hist,
        color="#1f77b4",
        linewidth=1.4,
        label="History (close)",
    )
    ax.plot(
        to_datenum(fut_t),
        med,
        color="#d62728",
        linewidth=1.8,
        linestyle="-",
        label="Forecast (median)",
    )
    if q10 is not None and q90 is not None:
        ax.fill_between(
            to_datenum(fut_t),
            q10,
            q90,
            color="#d62728",
            alpha=0.18,
            label="10th–90th percentile",
        )
    ax.axvline(
        to_datenum([base])[0],
        color="#555555",
        linewidth=0.8,
        linestyle="--",
        label="Today",
    )
    ax.set_title("TimesFM-3 forecast")
    ax.set_ylabel("Price")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fmt = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(fmt)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Plot saved to {args.out}")


if __name__ == "__main__":
    main()