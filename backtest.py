"""Rolling sweep backtest of TimesFM-3 on history, plus calibration stats.

Builds a series of overlapping windows ending at successive past bars,
forecasts horizon steps ahead, and compares the median/q10/q90 against the
bars that actually followed. Reports:
  - calibration: % of actuals below q10 / below median / below q90 (ideal 10/50/90)
  - coverage/hit-rate: % of actuals inside [q10, q90] (ideal 80%)
  - error stats: MAE, RMSE, bias of the median vs actual

CSV out: backtest_<name>.csv with one row per (window, horizon).

Usage:
    python backtest.py --csv data/eurusd_daily.csv --horizon 7 --samples 40
    python backtest.py --symbol EURUSD=X --tf D --horizon 7 --samples 40
"""
import argparse
import os

import numpy as np
from timesfm3 import ModelConfig, TimesFM3Evaluator

HORIZON_DEFAULT = {"D": 7, "240": 24}


def load_csv(path):
    closes = []
    for line in open(path).read().splitlines()[1:]:
        if not line.strip():
            continue
        closes.append(float(line.split(",")[4]))
    return np.array(closes, dtype=np.float32)


def evaluate(model, contexts, horizon):
    outs = list(
        model.predict_batch(
            contexts=contexts,
            horizon=horizon,
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
    )
    n = len(outs)
    med = np.zeros((horizon, n))
    q10 = np.zeros((horizon, n))
    q90 = np.zeros((horizon, n))
    for w, out in enumerate(outs):
        f = np.asarray(out.forecast).reshape(horizon)
        q = np.asarray(out.quantiles)  # (horizon, num_quantiles)
        if q.ndim == 3:  # multivariate path
            q = q[0]
        mid = q.shape[-1] // 2
        med[:, w] = f
        q10[:, w] = q[:, 0]
        q90[:, w] = q[:, -1]
    return med, q10, q90


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", help="local OHLCV CSV (fetch live data if not given)")
    ap.add_argument("--symbol", help="Yahoo symbol if fetching fresh data")
    ap.add_argument("--tf", default="D", choices=["D", "240"])
    ap.add_argument("--horizon", type=int, default=0, help="defaults 7 for D, 24 for 240")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=40, help="number of rolling windows")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.horizon == 0:
        args.horizon = HORIZON_DEFAULT[args.tf]

    csv_path = args.csv
    if not csv_path:
        import fetch_data

        csv_path = "data/_bt_tmp.csv"
        fetch_data.main_driver(args.symbol, args.tf, csv_path)

    closes = load_csv(csv_path)
    need = args.context + args.horizon
    if len(closes) < need:
        raise SystemExit(
            f"need {need} bars for windowing, have {len(closes)}. Use more history."
        )

    # window i forecasts from closes[i : i+context]; actual is closes[i+context : i+context+horizon]
    # Non-overlapping start indices, newest first so last bar is the most recent forecastable point.
    max_start = len(closes) - need
    starts = max_start - np.arange(args.samples) * args.horizon
    starts = starts[starts >= 0][::-1]  # oldest -> newest

    ctx = [closes[s : s + args.context] for s in starts]
    med, q10, q90 = evaluate(model_timesfm(args), ctx, args.horizon)

    rows = []
    inside = np.zeros(args.horizon, dtype=float)
    total = np.zeros(args.horizon, dtype=float)
    below_q10 = np.zeros(args.horizon, dtype=float)
    below_med = np.zeros(args.horizon, dtype=float)
    err_mae = np.zeros(args.horizon)
    err_rmse = np.zeros(args.horizon)
    bias = np.zeros(args.horizon)
    for w, s in enumerate(starts):
        actual = closes[s + args.context : s + args.context + args.horizon]
        for hh in range(args.horizon):
            a = actual[hh]
            m, lo, hi = med[hh, w], q10[hh, w], q90[hh, w]
            rows.append(
                (int(s + args.context), hh + 1, a, m, lo, hi, int(lo <= a <= hi))
            )
            total[hh] += 1
            inside[hh] += lo <= a <= hi
            below_q10[hh] += a < lo
            below_med[hh] += a < m
            e = abs(a - m)
            err_mae[hh] += e
            err_rmse[hh] += e * e
            bias[hh] += m - a

    name = os.path.splitext(os.path.basename(csv_path))[0]
    out_csv = args.out or f"data/backtest_{name}.csv"
    with open(out_csv, "w") as fh:
        fh.write("window_end,horizon,actual,median,q10,q90,inside\n")
        for r in rows:
            fh.write(",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in r))
            fh.write("\n")

    print("\n" + "=" * 76)
    print(f"Sweep backtest on {csv_path}  ({len(starts)} windows x {args.horizon}f)")
    print("=" * 76)
    print(f"{'h':>3} {'actual<q10':>9} {'actual<med':>9} {'actual<q90':>9} "
          f"{'inside':>7} {'MAE':>7} {'RMSE':>7} {'bias':>8}")
    for hh in range(args.horizon):
        n = total[hh]
        print(
            f"{hh+1:>3} {below_q10[hh]/n*100:8.1f}% {below_med[hh]/n*100:8.1f}% "
            f"{below_q90[hh]/n*100:8.1f}% {inside[hh]/n*100:6.1f}% "
            f"{err_mae[hh]/n:7.5f} {np.sqrt(err_rmse[hh]/n):7.5f} {bias[hh]/n:+8.5f}"
        )
    n = total[0] * args.horizon
    print("-" * 76)
    print(
        f"ALL {below_q10.sum()/n*100:8.1f}% {below_med.sum()/n*100:8.1f}% "
        f"{below_q90.sum()/n*100:8.1f}% {inside.sum()/n*100:6.1f}% "
        f"{err_mae.sum()/n:7.5f} {np.sqrt(err_rmse.sum()/n):7.5f} "
        f"{bias.sum()/n:+8.5f}"
    )
    print("Ideal: 10% / 50% / 90% / 80% inside.")
    print(f"\nCSV: {out_csv}")


def model_timesfm(args):
    cfg = ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=4,
        device=args.device,
    )
    return TimesFM3Evaluator(cfg)


if __name__ == "__main__":
    main()