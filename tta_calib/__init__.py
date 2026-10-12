"""tta-calib: shift-aware test-time adaptation with calibrated referral.

Novel method (SAC-TTA): estimate test-time shift severity from BatchNorm
statistic drift (label-free), blend base and TTA predictions with a
severity-gated weight, and refer uncertain cases with a temperature-scaled,
budget-matched threshold.
"""
from .shifts import SHIFT_TYPES, apply_shift, severity_grid
from .shift_estimator import ShiftEstimator
from .adaptive_tta import predict_adaptive, predict_naive_tta, AUGMENTATIONS
from .referral import (
    fit_temperature,
    referral_threshold_for_budget,
    ece_score,
    risk_coverage_auc,
    referral_stats,
)

__all__ = [
    "SHIFT_TYPES",
    "apply_shift",
    "severity_grid",
    "ShiftEstimator",
    "predict_adaptive",
    "predict_naive_tta",
    "AUGMENTATIONS",
    "fit_temperature",
    "referral_threshold_for_budget",
    "ece_score",
    "risk_coverage_auc",
    "referral_stats",
]
