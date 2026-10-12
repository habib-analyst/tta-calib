"""Full evaluation matrix for tta-calib.

For each dataset x shift type x severity level x method:
  methods: base | naive_tta | sac (shift-aware adaptive TTA)
For each method's probabilities:
  referral variants: none | fixed (tau=0.5, no temp scaling) | calibrated
                     (temperature scaling + budget-matched tau, B=0.15)

Metrics per cell: accuracy, ECE (raw + scaled), referral stats,
risk-coverage AUC, |referral_rate - budget|, per-batch estimated severity.

Outputs JSONL records to --out/results.jsonl (+ EVAL_DONE marker).
Resumable: skips conditions already present in results.jsonl (--resume).
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .model import DATASETS, get_model, load_split, make_loader
from .shifts import SHIFT_TYPES, apply_shift, severity_grid
from .shift_estimator import ShiftEstimator
from .adaptive_tta import predict_naive_tta, predict_adaptive
from .referral import (
    fit_temperature,
    apply_temperature,
    referral_threshold_for_budget,
    ece_score,
    risk_coverage_auc,
    referral_stats,
)

BUDGET = 0.15
FIXED_TAU = 0.5


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device)).cpu())
        all_y.append(y)
    return torch.cat(all_logits).numpy(), torch.cat(all_y).numpy()


def batched_predict(model, loader, device, method, estimator=None, sev_map=None):
    """Return (probs, labels, mean_severity). method in {base, naive_tta, sac}."""
    model.eval()
    probs_list, y_list, sevs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if method == "base":
                p = F.softmax(model(x), dim=1).cpu().numpy()
                sevs.append(0.0)
            elif method == "naive_tta":
                p = predict_naive_tta(model, x).cpu().numpy()
                sevs.append(1.0)
            elif method == "sac":
                s = estimator.severity(model, x)
                p, _ = predict_adaptive(model, x, s)
                p = p.cpu().numpy()
                sevs.append(s)
            else:
                raise ValueError(method)
            probs_list.append(p)
            y_list.append(y.numpy())
    return (
        np.concatenate(probs_list),
        np.concatenate(y_list),
        float(np.mean(sevs)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--model", required=True, help="path to best.pt")
    ap.add_argument("--root", default="~/.medmnist")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-conditions", type=int, default=0,
                    help="debug: cap number of shift conditions")
    args = ap.parse_args()
    device = torch.device("cpu")
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "results.jsonl")

    done = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["shift"], r["severity"], r["method"]))

    flag, num_classes, in_channels = DATASETS[args.dataset]
    model = get_model(num_classes, in_channels)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.to(device).eval()

    va_imgs, va_y = load_split(args.dataset, "val", root=args.root)
    te_imgs, te_y = load_split(args.dataset, "test", root=args.root)
    va_loader = make_loader(va_imgs, va_y, batch_size=args.batch)
    te_loader = make_loader(te_imgs, te_y, batch_size=args.batch)

    # --- validation: temperature + budget threshold (on clean val) ---
    va_logits, va_labels = collect_logits(model, va_loader, device)
    temperature = fit_temperature(va_logits, va_labels)
    va_probs = apply_temperature(va_logits, temperature)
    tau_budget = referral_threshold_for_budget(va_probs.max(axis=1), BUDGET)
    print(f"temperature={temperature:.3f} tau_budget={tau_budget:.3f}", flush=True)

    # --- shift estimators, one calibrator per shift type ---
    estimators = {}
    for st in SHIFT_TYPES:
        est = ShiftEstimator(model)

        def shift_fn(x, s01, _st=st):
            # map normalized severity -> nearest grid param
            grid = severity_grid(_st)
            idx = min(int(round(s01 * (len(grid) - 1))), len(grid) - 1)
            return apply_shift(x, _st, grid[idx][0])

        est.calibrate(model, va_loader, shift_fn, device=device, n_batches=6)
        estimators[st] = est
    print("estimators calibrated", flush=True)

    # --- eval matrix ---
    conditions = []
    for st in SHIFT_TYPES:
        for param, sev in severity_grid(st):
            conditions.append((st, param, sev))
    if args.max_conditions:
        conditions = conditions[: args.max_conditions]

    n_new = 0
    for st, param, sev in conditions:
        te_shifted = apply_shift(
            torch.from_numpy(te_imgs), st, param
        ).numpy()
        te_shift_loader = make_loader(te_shifted, te_y, batch_size=args.batch)
        for method in ["base", "naive_tta", "sac"]:
            key = (st, sev, method)
            if key in done:
                continue
            probs, labels, mean_sev = batched_predict(
                model, te_shift_loader, device, method,
                estimator=estimators[st] if method == "sac" else None,
            )
            acc = float((probs.argmax(1) == labels).mean())
            ece_raw = ece_score(probs, labels)
            probs_t = apply_temperature(
                torch.from_numpy(
                    np.log(np.clip(probs, 1e-12, 1.0))
                ),
                temperature,
            )
            # note: probs are already softmaxed; re-derive "logits" approx is
            # unnecessary — temperature was fit on true logits; here we apply
            # the fitted T to log-probs which is equivalent up to a constant.
            ece_cal = ece_score(probs_t, labels)
            rc_auc = risk_coverage_auc(probs_t, labels)

            rec = {
                "dataset": args.dataset,
                "shift": st,
                "param": param,
                "severity": sev,
                "method": method,
                "n": len(labels),
                "acc": acc,
                "ece_raw": ece_raw,
                "ece_cal": ece_cal,
                "risk_coverage_auc": rc_auc,
                "mean_est_severity": mean_sev,
                "temperature": temperature,
            }
            # referral variants on temperature-scaled probs
            for variant, tau in [
                ("fixed", FIXED_TAU),
                ("calibrated", tau_budget),
            ]:
                rs = referral_stats(probs_t, labels, tau)
                rs["budget_deviation"] = abs(rs["referral_rate"] - BUDGET)
                for k, v in rs.items():
                    rec[f"referral_{variant}_{k}"] = v
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n_new += 1
            print(
                f"{st} sev={sev:.2f} {method}: acc={acc:.4f} "
                f"ece={ece_cal:.4f} rc_auc={rc_auc:.4f} "
                f"ref_cal={rec['referral_calibrated_referral_rate']:.3f}",
                flush=True,
            )
    open(os.path.join(args.out, "EVAL_DONE"), "w").write("done\n")
    print(f"DONE: {n_new} new records -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
