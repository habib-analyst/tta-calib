# SAC-TTA: Shift-Aware Test-Time Adaptation with Calibrated Referral

**Habib Ur Rehman** — independent research, October 2026

## 1. Problem

Two failure modes of test-time adaptation under distribution shift, both
measured in our MedMNIST study (medmodel-discovery, Exp-C):

1. **Fixed-strength TTA can hurt.** Naive test-time augmentation beat a
   gated-consensus variant under noise shift (0.6058 vs 0.5064 accuracy on
   pneumonia), but any fixed TTA policy risks degrading clean-data
   performance — there is no free adaptation strength.
2. **Referral thresholds miscalibrate under shift.** A fixed uncertainty
   threshold produced referral rates of 0.43/0.69 against a 0.15 target —
   the model's "I don't know" signal does not track its true error rate
   once the distribution moves.

## 2. Method: SAC-TTA

One label-free signal drives both adaptation and referral: **BatchNorm
statistic drift**. A trained model's BN running stats summarize the training
feature distribution; at test time, per-batch feature statistics drift away
from them in proportion to shift severity.

**Shift-severity estimator.** For each BN layer, compute the normalized drift
of the test batch's channel means/variances against the stored running stats,
averaged over layers. An isotonic-regression calibrator (fit on validation
batches with known injected severities) maps raw drift to severity in
[0, 1]. No labels needed; one forward pass with hooks.

**Adaptive TTA.** Blend base and augmented-consensus predictions with the
severity as gate:

```
p = (1 − α) · p_base + α · p_tta,   α = estimated severity
```

Clean input (α ≈ 0) → trust the base model, skipping TTA cost and its
degradation risk. Heavy shift (α ≈ 1) → full augmented consensus.
Augmentations are cheap integer ops (flip / 1-px rolls).

**Calibrated referral.** Temperature scaling (NLL-minimizing T on
validation) + threshold set to the referral-budget quantile of validation
confidences. The referred fraction tracks the budget even as the
distribution shifts; a fixed threshold drifts.

## 3. Experimental design

- **Data:** PneumoniaMNIST (2-class), DermaMNIST (7-class),
  BloodMNIST (8-class) @ 28px; ResNet-18 (CIFAR-style stem), CPU-trained.
- **Shifts:** Gaussian noise {0, 0.05, 0.1, 0.2, 0.3}, Gaussian blur
  {0, 0.5, 1.0, 1.5, 2.0}, contrast {1.0, 0.7, 0.5, 0.3}, brightness
  {0, 0.1, 0.2, 0.3} — 18 conditions total (identity-param levels are
  clean-equivalent).
- **Baselines:** base (no TTA), naive TTA (fixed α = 1).
- **Referral variants:** none, fixed threshold (τ = 0.5, unscaled),
  calibrated (temperature + 15% budget).
- **Metrics:** accuracy (clean + per-severity), ECE, referral-rate
  calibration (|referred fraction − budget|), referral precision (error rate
  among referred), risk-coverage AUC, estimator quality (Spearman ρ between
  estimated and injected severity).

## 4. Results

All numbers measured; tables from `results/summary.json` (54 eval records
per dataset: 18 conditions × 3 methods). Pre-registered success criteria
were set before results existed; verdicts below judge them as written.

### 4.1 Estimator quality — criterion FAIL

Spearman ρ between estimated and injected severity, per shift family:

| dataset     | noise | blur  | contrast | brightness |
|-------------|------:|------:|---------:|-----------:|
| pneumonia   | 0.172 | 0.108 | 0.200    | 0.112      |
| derma       | 0.092 | 0.000 | −0.045   | 0.045      |
| blood       | 0.113 | 0.113 | 0.183    | 0.183      |

Pre-registered bar: ρ > 0.7. Measured: 0.09–0.20. **Fail, decisively.**
BN-statistic drift does not track injected severity on these models and
data. The "shift-aware" mechanism claim is dead: with a useless severity
signal, SAC's α gate hovers near a constant, and adaptive TTA collapses to
a fixed blend — which is exactly why SAC ≈ naive TTA everywhere below.

### 4.2 Accuracy: clean + under shift

| dataset   | acc_clean base / naive / sac | Δ clean sac−base | acc_shift base / naive / sac |
|-----------|------------------------------|------------------|------------------------------|
| pneumonia | 0.8317 / 0.8333 / 0.8297     | −0.20pp          | 0.6668 / 0.6631 / 0.6647     |
| derma     | 0.6918 / 0.6943 / 0.6931     | +0.13pp          | 0.6637 / 0.6634 / 0.6631     |
| blood     | 0.8948 / 0.9021 / 0.8951     | +0.03pp          | 0.4499 / 0.4497 / 0.4492     |

- **Criterion (3): clean degradation sac vs base ≤ 0.5pp — PASS** on all
  three (worst: pneumonia −0.20pp). Adaptive gating, even with a dead
  severity signal, costs nothing on clean data.
- **Criterion (2): sac acc_shift ≥ naive TTA — not meaningfully met.**
  Pneumonia: sac +0.16pp over naive; derma: naive +0.03pp; blood:
  naive +0.05pp. A wash within noise; no dataset shows a real gain.

### 4.3 Calibration and risk–coverage under shift

| dataset   | ECE_shift base / naive / sac | RC-AUC_shift base / naive / sac |
|-----------|------------------------------|---------------------------------|
| pneumonia | 0.2717 / 0.2627 / 0.2658      | 0.1863 / 0.1887 / 0.1879        |
| derma     | 0.0647 / 0.0654 / 0.0665      | 0.1374 / 0.1369 / 0.1369        |
| blood     | 0.4390 / 0.4310 / 0.4309      | 0.4678 / 0.4678 / 0.4680        |

TTA (both variants) slightly *improves* ECE under shift vs base on
pneumonia (−0.006 to −0.009) and blood (−0.008); derma is already
well-calibrated and base is marginally best there. Risk–coverage AUC
moves ≤ 0.0023 absolute between methods — essentially unmoved. Accuracy
doesn't shift, calibration barely shifts: TTA changes little here.

### 4.4 Referral: budget tracking and error concentration

Target referral budget 15%; budget deviation = |referred fraction − 0.15|
under shift. Error rate among referred vs overall shift error rate
(1 − acc_shift_mean, SAC) shows whether referrals concentrate errors.

| dataset   | budget_dev base / naive / sac | fixed-threshold dev | err among referred (sac) | overall shift err (sac) |
|-----------|-------------------------------|---------------------|--------------------------|-------------------------|
| pneumonia | 0.2057 / 0.2455 / **0.2274**  | **0.1500**          | 0.538                    | 0.335                   |
| derma     | 0.0651 / 0.0649 / **0.0644**  | 0.1897              | 0.700                    | 0.337                   |
| blood     | 0.0616 / 0.0785 / **0.0706**  | 0.1136              | 0.686                    | 0.551                   |

- **Criterion (4): sac budget-dev < fixed-threshold — mixed, 2/3.**
  SAC tracks the budget better than a fixed τ = 0.5 on derma
  (0.0644 vs 0.1897) and blood (0.0706 vs 0.1136); fixed threshold
  wins on pneumonia (0.1500 vs 0.2274).
- **Referrals concentrate errors.** Error rate among referred (0.54–0.70)
  is 1.25×–2.1× the overall shift error rate on all three datasets.
  The referral *criterion* has genuine signal — it flags where the
  model is actually wrong — even though no method's accuracy moves.

### 4.5 Verdict table

| # | pre-registered criterion | verdict |
|---|--------------------------|---------|
| 1 | shift-severity estimator ρ > 0.7 | **FAIL** (ρ = 0.09–0.20) |
| 2 | sac acc_shift ≥ naive TTA | not meaningfully met (wash ±0.16pp) |
| 3 | clean degradation sac vs base ≤ 0.5pp | **PASS** (worst −0.20pp) |
| 4 | sac budget-dev < fixed threshold | mixed, 2/3 (pneumonia fixed wins) |

**Headline:** mixed/negative. The core shift-awareness mechanism did not
validate — estimator failure (4.1) explains why SAC ≈ naive TTA. The
surviving contribution is temperature-scaled referral targeting: it tracks
the referral budget better than a fixed threshold on 2/3 datasets with
clean accuracy preserved, and referrals concentrate errors 1.25–2.1× —
errors are better *flagged* even though they are not *fixed*.
A publishable negative in the foundry's negative-result discipline.

## 5. Limitations & honest notes

- **Estimator failure (this study's central negative result).**
  The BN-drift severity signal does not track injected severity
  (ρ = 0.09–0.20 vs pre-registered bar > 0.7, §4.1). The "shift-aware"
  half of SAC-TTA is disproven on ResNet-18 @ 28px; what remains is
  the calibrated-referral half.
- **Referrals concentrate errors but accuracy does not move.**
  Flagging errors is not fixing them; clinical value would require
  the referred cases to get a better downstream decision, untested here.
- Shift-type-specific calibrators: the severity mapping is fit per shift
  family. A deployment would need a shift-type-agnostic estimator —
  listed as future work, not claimed.
- Synthetic shifts only; real-world scanner/protocol shift is the true
  target and is not measured here.
- 28px MedMNIST; scaling behavior to full-resolution images untested.
- 18 conditions × 3 methods per dataset is a small eval matrix; severity
  grids are coarse (4–5 levels per family).

## 6. Reproducibility

All code, seeds, and the exact eval matrix are in the repo. Training is
resumable (`--resume`); eval skips completed conditions. Total compute:
CPU-only, $0.
