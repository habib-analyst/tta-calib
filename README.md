[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)] [![GitHub stars](https://img.shields.io/github/stars/habib-analyst/tta-calib.svg)](https://github.com/habib-analyst/tta-calib)

# tta-calib — Shift-Aware Test-Time Adaptation with Calibrated Referral

Fixed test-time adaptation (TTA) can **hurt** on clean data, and fixed
uncertainty thresholds **miscalibrate** under shift (our medmodel-discovery
Exp-C measured referral rates of 0.43/0.69 against a 0.15 target, and naive
TTA beating gated consensus under noise: 0.6058 vs 0.5064).

**SAC-TTA** fixes both with one label-free signal — BatchNorm statistic drift:

1. **Shift-severity estimator**: normalized drift of test-batch feature
   statistics vs training BN running stats → calibrated severity in [0,1]
   (isotonic regression on validation shifts).
2. **Adaptive TTA**: `p = (1-α)·p_base + α·p_tta` with `α = severity`.
   Clean input → trust the base model; heavy shift → full augmented consensus.
3. **Calibrated referral**: temperature scaling + threshold matched to a
   referral budget on validation, so the referred fraction tracks the budget
   even under shift.

## Results (measured, CPU-only)

See `results/summary.json` and `papers/PAPER.md` for the full tables.
Headlines (ResNet-18, MedMNIST-28, synthetic noise/blur/contrast/brightness;
full verdicts in `papers/PAPER.md` §4 — this is a mixed/negative result):

- **Shift estimator FAILS the pre-registered bar**: Spearman ρ = 0.09–0.20
  vs bar > 0.7. BN-statistic drift does not track shift severity at
  28px; the "shift-aware" mechanism claim is dead (which explains why
  adaptive TTA ≈ naive TTA everywhere).
- SAC vs naive TTA under shift is a wash within noise (pneumonia
  sac +0.16pp, derma/blood naive +0.03/+0.05pp); clean accuracy is
  preserved (worst −0.20pp).
- **Calibrated referral is the surviving contribution**: temperature
  scaling + budget-matched threshold tracks the 15% referral budget
  better than a fixed threshold on 2/3 datasets (derma, blood), and
  referred cases concentrate errors 1.25–2.1× above base rates —
  errors get better *flagged* even though accuracy doesn't move.
- Risk-coverage AUC and ECE reported per method; TTA slightly improves
  ECE under shift vs base on pneumonia and blood.

## Layout

- `tta_calib/shifts.py` — synthetic shifts at graded severities
- `tta_calib/shift_estimator.py` — BN-drift severity estimator + calibration
- `tta_calib/adaptive_tta.py` — severity-gated TTA blending
- `tta_calib/referral.py` — temperature scaling, budget referral, ECE, RC-AUC
- `tta_calib/train.py` / `evaluate.py` — resumable training + eval matrix
- `run_experiments.py` — full pipeline (train → eval → summarize)
- `tests/` — unit tests (`pytest`)

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# MedMNIST npz files expected at ~/.medmnist (or pass --root)
setsid nohup python3 run_experiments.py > run.log 2>&1 < /dev/null &
```

## License

MIT
