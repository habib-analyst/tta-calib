"""Orchestrate the full tta-calib experiment pipeline (CPU).

Stages:
  1. Train ResNet-18 per dataset (pneumonia, derma, blood) — skips finished.
  2. Evaluate the shift x severity x method matrix per dataset — resumable.
  3. Summarize results -> results/summary.json.

Launch detached per the charter (daemon restarts kill workers):
  setsid nohup python3 run_experiments.py > run.log 2>&1 < /dev/null &
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["pneumoniamnist", "dermamnist", "bloodmnist"]
EPOCHS = {"pneumoniamnist": 6, "dermamnist": 8, "bloodmnist": 8}


def sh(cmd):
    print("+ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}")


def main():
    py = os.path.join(HERE, ".venv", "bin", "python")
    results_root = os.path.join(HERE, "results")
    os.makedirs(results_root, exist_ok=True)

    for ds in DATASETS:
        model_dir = os.path.join(HERE, "models", ds)
        done_marker = os.path.join(model_dir, "TRAIN_DONE")
        if not os.path.exists(done_marker):
            sh([
                py, "-m", "tta_calib.train",
                "--dataset", ds,
                "--epochs", str(EPOCHS[ds]),
                "--out", model_dir,
                "--resume",
            ])
        else:
            print(f"{ds}: training already done, skipping", flush=True)

        eval_dir = os.path.join(results_root, ds)
        eval_marker = os.path.join(eval_dir, "EVAL_DONE")
        if not os.path.exists(eval_marker):
            sh([
                py, "-m", "tta_calib.evaluate",
                "--dataset", ds,
                "--model", os.path.join(model_dir, "best.pt"),
                "--out", eval_dir,
                "--resume",
            ])
        else:
            print(f"{ds}: eval already done, skipping", flush=True)

    summarize(results_root)
    open(os.path.join(results_root, "PIPELINE_DONE"), "w").write("done\n")
    print("PIPELINE DONE", flush=True)


def summarize(results_root):
    """Aggregate headline metrics per dataset/method across severities."""
    import numpy as np

    summary = {}
    for ds in DATASETS:
        path = os.path.join(results_root, ds, "results.jsonl")
        recs = [json.loads(l) for l in open(path)]
        # shift-estimator quality: spearman(true sev, est sev) per shift
        est_rows = {}
        for r in recs:
            est_rows.setdefault(r["shift"], []).append(
                (r["severity"], r["mean_est_severity"])
            )
        from scipy.stats import spearmanr

        est_quality = {
            st: float(spearmanr([a for a, _ in v], [b for _, b in v])[0])
            for st, v in est_rows.items()
        }
        per_method = {}
        for m in ["base", "naive_tta", "sac"]:
            sub = [r for r in recs if r["method"] == m]
            clean = [r for r in sub if r["severity"] == 0.0]
            shifted = [r for r in sub if r["severity"] > 0.0]
            per_method[m] = {
                "acc_clean": float(np.mean([r["acc"] for r in clean])),
                "acc_shift_mean": float(np.mean([r["acc"] for r in shifted])),
                "ece_cal_shift_mean": float(
                    np.mean([r["ece_cal"] for r in shifted])
                ),
                "rc_auc_shift_mean": float(
                    np.mean([r["risk_coverage_auc"] for r in shifted])
                ),
                "referral_budget_dev_shift_mean": float(
                    np.mean(
                        [r["referral_calibrated_budget_deviation"] for r in shifted]
                    )
                ),
                "referral_fixed_budget_dev_shift_mean": float(
                    np.mean([r["referral_fixed_budget_deviation"] for r in shifted])
                ),
                "err_among_referred_shift_mean": float(
                    np.nanmean(
                        [r["referral_calibrated_err_among_referred"] for r in shifted]
                    )
                ),
            }
        summary[ds] = {"estimator_spearman": est_quality, "methods": per_method}
    with open(os.path.join(results_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1)[:2000], flush=True)


if __name__ == "__main__":
    main()
