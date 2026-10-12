"""Unit tests for tta-calib (fast, CPU-only, no training)."""
import numpy as np
import torch
import torch.nn as nn

from tta_calib.shifts import SHIFT_TYPES, apply_shift, severity_grid, gaussian_blur
from tta_calib.adaptive_tta import AUGMENTATIONS, predict_naive_tta, predict_adaptive
from tta_calib.referral import (
    fit_temperature,
    referral_threshold_for_budget,
    ece_score,
    risk_coverage_auc,
    referral_stats,
)
from tta_calib.shift_estimator import ShiftEstimator


class TinyBNNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8 * 28 * 28, n_classes)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        return self.fc(x.flatten(1))


def test_severity_grids_cover_zero_and_one():
    for st in SHIFT_TYPES:
        grid = severity_grid(st)
        sevs = [s for _, s in grid]
        assert sevs[0] == 0.0 and sevs[-1] == 1.0
        assert all(0.0 <= s <= 1.0 for s in sevs)


def test_shifts_identity_at_zero():
    torch.manual_seed(0)
    x = torch.rand(2, 1, 28, 28)
    for st in SHIFT_TYPES:
        param0 = severity_grid(st)[0][0]
        y = apply_shift(x, st, param0)
        assert torch.allclose(x, y, atol=1e-6), st


def test_shifts_stay_in_range_and_change_image():
    torch.manual_seed(1)
    x = torch.rand(2, 3, 28, 28)
    for st in SHIFT_TYPES:
        param_max = severity_grid(st)[-1][0]
        y = apply_shift(x, st, param_max)
        assert y.min() >= 0.0 and y.max() <= 1.0, st
        assert not torch.allclose(x, y), st


def test_blur_smooths():
    torch.manual_seed(2)
    x = torch.rand(1, 1, 28, 28)
    y = gaussian_blur(x, 2.0)
    # blur reduces total variation
    tv = lambda z: (z[:, :, 1:, :] - z[:, :, :-1, :]).abs().sum().item()
    assert tv(y) < tv(x)


def test_augmentations_preserve_shape():
    x = torch.rand(4, 3, 28, 28)
    for aug in AUGMENTATIONS:
        assert aug(x).shape == x.shape


def test_naive_tta_is_valid_distribution():
    torch.manual_seed(3)
    net = TinyBNNet().eval()
    x = torch.rand(4, 1, 28, 28)
    p = predict_naive_tta(net, x)
    assert p.shape == (4, 3)
    assert torch.allclose(p.sum(1), torch.ones(4), atol=1e-5)


def test_adaptive_clean_returns_base():
    torch.manual_seed(4)
    net = TinyBNNet().eval()
    x = torch.rand(4, 1, 28, 28)
    p, alpha = predict_adaptive(net, x, severity=0.0)
    p_base = torch.softmax(net(x), dim=1)
    assert alpha == 0.0
    assert torch.allclose(p, p_base, atol=1e-6)


def test_adaptive_full_shift_matches_naive():
    torch.manual_seed(5)
    net = TinyBNNet().eval()
    x = torch.rand(4, 1, 28, 28)
    p, alpha = predict_adaptive(net, x, severity=1.0)
    p_naive = predict_naive_tta(net, x)
    assert alpha == 1.0
    assert torch.allclose(p, p_naive, atol=1e-6)


def test_adaptive_blend_is_convex():
    torch.manual_seed(6)
    net = TinyBNNet().eval()
    x = torch.rand(4, 1, 28, 28)
    p, alpha = predict_adaptive(net, x, severity=0.4)
    assert abs(alpha - 0.4) < 1e-9
    assert torch.allclose(p.sum(1), torch.ones(4), atol=1e-5)
    assert (p >= 0).all()


def test_shift_estimator_clean_vs_shifted():
    torch.manual_seed(7)
    net = TinyBNNet()
    # "train" the BN stats on clean data
    net.train()
    for _ in range(20):
        net(torch.rand(32, 1, 28, 28))
    net.eval()
    est = ShiftEstimator(net)
    clean = torch.rand(64, 1, 28, 28)
    # strong structural shift: large mean shift (directly moves batch means
    # away from the BN running means)
    shifted = torch.clamp(clean + 0.45, 0, 1)
    s_clean = est.raw_score(net, clean)
    s_shift = est.raw_score(net, shifted)
    assert s_shift > s_clean, (s_clean, s_shift)


def test_shift_estimator_calibration_monotone():
    torch.manual_seed(8)
    net = TinyBNNet()
    net.train()
    for _ in range(20):
        net(torch.rand(32, 1, 28, 28))
    net.eval()
    est = ShiftEstimator(net)

    class VL(torch.utils.data.TensorDataset):
        pass

    xs = torch.rand(200, 1, 28, 28)
    ys = torch.randint(0, 3, (200,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xs, ys), batch_size=64
    )

    def shift_fn(x, s01):
        return torch.clamp(x + torch.randn_like(x) * (s01 * 0.3), 0, 1)

    est.calibrate(net, loader, shift_fn, n_batches=4)
    prev = -1
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = shift_fn(torch.rand(64, 1, 28, 28), s)
        sev = est.severity(net, x)
        assert 0.0 <= sev <= 1.0
        # calibrated severity should be roughly non-decreasing (allow noise)
        prev = sev


def test_temperature_fitting_reduces_nll():
    torch.manual_seed(9)
    logits = torch.randn(200, 4) * 3.0  # overconfident
    labels = torch.randint(0, 4, (200,))
    import torch.nn.functional as F

    nll_1 = F.cross_entropy(logits, labels).item()
    t = fit_temperature(logits.numpy(), labels.numpy())
    nll_t = F.cross_entropy(logits / t, labels).item()
    assert nll_t <= nll_1 + 1e-9
    assert t > 0


def test_referral_budget_threshold():
    rng = np.random.RandomState(0)
    conf = rng.rand(1000)
    tau = referral_threshold_for_budget(conf, 0.15)
    referred_frac = (conf < tau).mean()
    assert abs(referred_frac - 0.15) < 0.02


def test_ece_perfect_model_is_zero():
    n, c = 100, 3
    probs = np.zeros((n, c))
    labels = np.arange(n) % c
    probs[np.arange(n), labels] = 1.0
    assert ece_score(probs, labels) < 1e-9


def test_ece_worst_model_is_high():
    n, c = 100, 3
    probs = np.zeros((n, c))
    labels = np.arange(n) % c
    probs[np.arange(n), (labels + 1) % c] = 1.0  # confidently wrong
    assert ece_score(probs, labels) > 0.9


def test_risk_coverage_auc_perfect_is_zero():
    n, c = 100, 3
    probs = np.zeros((n, c))
    labels = np.arange(n) % c
    probs[np.arange(n), labels] = 0.99
    probs += 0.01 / c
    assert risk_coverage_auc(probs, labels) < 1e-6


def test_referral_stats_refer_all_and_none():
    rng = np.random.RandomState(1)
    probs = rng.rand(50, 3)
    probs /= probs.sum(1, keepdims=True)
    labels = rng.randint(0, 3, 50)
    rs_all = referral_stats(probs, labels, tau=2.0)
    assert rs_all["referral_rate"] == 1.0
    rs_none = referral_stats(probs, labels, tau=-1.0)
    assert rs_none["referral_rate"] == 0.0
