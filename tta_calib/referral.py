"""Calibrated referral: temperature scaling + budget-matched threshold.

Referral rule: refer a case to a human expert if max softmax prob < tau.
Two ingredients make it *calibrated*:

1. Temperature scaling (Guo et al.): fit T on validation logits by NLL
   minimization so confidences approximate true correctness likelihood.
2. Budget matching: choose tau as the B-quantile of validation confidences,
   so the *expected* referral rate equals the budget B. A fixed threshold
   (e.g. 0.5) drifts under shift; the budget-matched one tracks.

Metrics: ECE, referral precision (error rate among referred), referral-rate
calibration (|test referral rate - B|), and risk-coverage AUC.
"""
import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits, labels, device="cpu"):
    """Grid-search temperature minimizing validation NLL. Returns float T."""
    logits = torch.as_tensor(logits, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    best_t, best_nll = 1.0, float("inf")
    for t in np.concatenate(
        [np.linspace(0.2, 1.0, 9), np.linspace(1.1, 5.0, 20)]
    ):
        nll = F.cross_entropy(logits / t, labels).item()
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def apply_temperature(logits, temperature):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    return F.softmax(logits / temperature, dim=1).numpy()


def referral_threshold_for_budget(confidences, budget):
    """Threshold tau such that fraction `budget` of validation cases refer.

    confidences: array of max softmax probs on validation.
    Refer if confidence < tau.
    """
    confidences = np.asarray(confidences)
    if not 0 < budget < 1:
        raise ValueError("budget must be in (0, 1)")
    return float(np.quantile(confidences, budget))


def ece_score(probs, labels, n_bins=15):
    """Expected Calibration Error."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def risk_coverage_auc(probs, labels, n_points=100):
    """Area under the risk-coverage curve (lower = better).

    Sort by confidence desc; at coverage c (top-c fraction kept), risk =
    error rate among kept. AUC over c in (0, 1].
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    order = np.argsort(-conf)
    correct = correct[order]
    coverages = np.linspace(1 / n_points, 1.0, n_points)
    risks = []
    for c in coverages:
        k = max(int(c * len(correct)), 1)
        risks.append(1.0 - correct[:k].mean())
    return float(np.trapezoid(risks, coverages))


def referral_stats(probs, labels, tau):
    """Referral diagnostics at threshold tau (refer if max_prob < tau)."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    referred = conf < tau
    err = (pred != labels)
    n = len(labels)
    out = {
        "referral_rate": float(referred.mean()),
        "n_referred": int(referred.sum()),
        "n_total": n,
    }
    if referred.sum() > 0:
        out["err_among_referred"] = float(err[referred].mean())
    else:
        out["err_among_referred"] = float("nan")
    if (~referred).sum() > 0:
        out["err_among_kept"] = float(err[~referred].mean())
    else:
        out["err_among_kept"] = float("nan")
    out["overall_err"] = float(err.mean())
    return out
