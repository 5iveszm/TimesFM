"""TimesFM-3 forex forecasting from TradingView OHLCV CSVs.

Feeds close prices (optionally multi-variate: close+high+low) into Google's
TimesFM-3 foundation model and prints point + quantile forecasts.

Zero-shot: no training, works on any CSV matching the pull.js schema:
    time,open,high,low,close,volume   (time = unix seconds)

Usage:
    python forecast.py --csv data/eurusd_daily.csv --horizon 30 --context 1024
    python forecast.py --csv data/eurusd_4h.csv --horizon 48 --multivariate

Requires:  pip install timesfm[torch]   (PyTorch backend, Python >= 3.10)
"""
import argparse
import math
import sys

import numpy as np

try:
    from timesfm3 import TimesFM3Evaluator, ModelConfig
except ImportError:
    print(
        "timesfm3 not installed. Run:  pip install 'timesfm[torch]'",
        file=sys.stderr,
    )
    sys.exit(1)


def load_csv(path):
    """Read a pull.js CSV into (times_seconds, prices dict)."""
    times, o, h, l, c, v = [], [], [], [], [], []
    with open(path) as fh:
        next(fh)  # header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            times.append(int(parts[0].split(".")[0]))
            o.append(float(parts[1]))
            h.append(float(parts[2]))
            l.append(float(parts[3]))
            c.append(float(parts[4]))
            v.append(float(parts[5]))
    prices = {"open": o, "high": h, "low": l, "close": c, "volume": v}
    return np.array(times, dtype=np.int64), prices


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="OHLCV CSV from pull.js")
    ap.add_argument("--horizon", type=int, default=30, help="forecast length in bars")
    ap.add_argument(
        "--context",
        type=int,
        default=1024,
        help="number of history bars to feed (default 1024)",
    )
    ap.add_argument(
        "--multivariate",
        action="store_true",
        help="forecast close+high+low jointly (3 variates)",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        help="cuda, cpu, or mps (default: cpu)",
    )
    ap.add_argument(
        "--backtest-len",
        type=int,
        default=0,
        help="if >0, hold out this many bars from the end and report error vs actual",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    times, prices = load_csv(args.csv)
    n = len(prices["close"])
    print(f"Loaded {n} bars from {args.csv} (last bar date shown in columns).")

    # Build the 2D target matrix: rows = variates, cols = time steps.
    if args.multivariate:
        target = np.stack([prices["close"], prices["high"], prices["low"]])
        print("Multivariate forecast on variates: close, high, low")
    else:
        target = np.array(prices["close"])[None, :]
        print("Univariate forecast on close prices")

    if n < args.context + args.horizon + (args.backtest_len or 1):
        print(
            f"Not enough bars: have {n}, need context({args.context}) + horizon "
            f"({args.horizon}) + held-out({args.backtest_len})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Held-out slice for a quick sanity check of the forecast.
    usable = n - args.backtest_len
    if args.backtest_len:
        actual = target[:, usable:]
        target = target[:, :usable]
        print(f"Holding out last {args.backtest_len} bars for backtest.")

    context = target[:, -args.context :]

    config = ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=args.batch_size,
        device=args.device,
    )
    print("Loading TimesFM-3 (first run downloads ~1GB weights)...")
    forecaster = TimesFM3Evaluator(config)

    print(
        f"Forecasting {args.horizon} bars from a {args.context}-bar context "
        f"({context.shape[0]} variates)..."
    )
    outputs = list(
        forecaster.predict_batch(
            contexts=[context],
            horizon=args.horizon,
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
    )
    out = outputs[0]
    forecast = out.forecast  # (n_variates, horizon)
    quantiles = out.quantiles  # (n_variates, horizon, n_quantiles)

    # Median is at index 4 (from model card). Print every quantile once per
    # variate as a small table, plus the median series.
    q_keys = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    med_idx = q_keys.index(50)

    last_time = int(times[-1] if args.backtest_len == 0 else times[usable - 1])
    names = ["close", "high", "low"] if args.multivariate else ["close"]

    results = {}
    for vi, name in enumerate(names):
        series = {}
        last_close = prices["close"][usable - 1] if args.backtest_len else prices["close"][-1]
        base_time = last_time
        for hh in range(args.horizon):
            row = {
                "time": base_time,
                f"q10": quantiles[vi, hh, 0],
                f"median": quantiles[vi, hh, med_idx],
                f"q90": quantiles[vi, hh, -1],
            }
            series[hh] = row
            # Advance by the average bar spacing as an approximate time label.
            if len(times) >= args.context + 2:
                dt = times[usable - 1] - times[usable - args.context]
                dt = int(dt / (args.context - 1))
                base_time += dt
        results[name] = series

    print("\n" + "=" * 70)
    print(f"{names[0].upper()} — median forecast for next {args.horizon} bars")
    print("=" * 70)
    mid = results[names[0]]
    for hh in range(args.horizon):
        r = mid[hh]
        bar = hh + 1
        label = f"t+{bar}"
        print(
            f"{label:<6} median={r['median']:.5f}  q10={r['q10']:.5f}  "
            f"q90={r['q90']:.5f}"
        )

    if args.multivariate:
        print("\n--- High / Low medians ---")
        for name in ("high", "low"):
            for hh in range(args.horizon):
                r = results[name][hh]
                print(f"t+{hh+1:<4} {name:<5} median={r['median']:.5f}")

    if args.backtest_len:
        print("\n--- Backtest vs actual (median) ---")
        for hh in range(args.backtest_len):
            f_ = forecast[0, hh]
            a_ = actual[0, hh]
            print(f"t+{hh+1:<4} forecast={f_:.5f} actual={a_:.5f}")

    # Also dump machine-readable CSV.
    out_csv = args.csv.replace(".csv", f"_forecast_{args.horizon}.csv")
    with open(out_csv, "w") as fh:
        qcols = []
        for name in names:
            qcols += [f"{name}_q10", f"{name}_median", f"{name}_q90"]
        fh.write("horizon," + ",".join(qcols) + "\n")
        for hh in range(args.horizon):
            row = [str(hh + 1)]
            for name in names:
                r = results[name][hh]
                row.append(f"{r['q10']:.8f}")
                row.append(f"{r['median']:.8f}")
                row.append(f"{r['q90']:.8f}")
            fh.write(",".join(row) + "\n")
    print(f"\nForecast written to {out_csv}")


if __name__ == "__main__":
    main()