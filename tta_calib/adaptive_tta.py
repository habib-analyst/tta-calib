"""Severity-gated test-time adaptation.

Naive TTA always averages augmented views (alpha=1), which can hurt on clean
data. SAC-TTA blends the base prediction with the augmented consensus using
the estimated shift severity as the gate:

    p = (1 - alpha) * p_base + alpha * p_tta,   alpha = severity in [0,1]

Clean input  -> alpha ~ 0 -> trust the base model (no TTA degradation).
Heavy shift  -> alpha ~ 1 -> full augmented consensus.

Augmentations are cheap integer ops (flip / 1px rolls), no extra deps.
"""
import torch
import torch.nn.functional as F


def _hflip(x):
    return torch.flip(x, dims=(-1,))


def _roll_up(x):
    return torch.roll(x, shifts=1, dims=(-2,))


def _roll_down(x):
    return torch.roll(x, shifts=-1, dims=(-2,))


def _roll_left(x):
    return torch.roll(x, shifts=1, dims=(-1,))


def _roll_right(x):
    return torch.roll(x, shifts=-1, dims=(-1,))


AUGMENTATIONS = [_hflip, _roll_up, _roll_down, _roll_left, _roll_right]


@torch.no_grad()
def _softmax_logits(model, x):
    return F.softmax(model(x), dim=1)


@torch.no_grad()
def predict_naive_tta(model, x):
    """Fixed-strength TTA: mean of base + all augmented softmaxes (alpha=1)."""
    probs = [_softmax_logits(model, x)]
    for aug in AUGMENTATIONS:
        probs.append(_softmax_logits(model, aug(x)))
    return torch.stack(probs).mean(dim=0)


@torch.no_grad()
def predict_adaptive(model, x, severity, skip_threshold=0.05):
    """Severity-gated blend of base and TTA consensus.

    severity: float in [0, 1] from ShiftEstimator. Below skip_threshold the
    base prediction is returned directly (saves compute on clean data).
    Returns (probs, alpha_used).
    """
    alpha = float(min(max(severity, 0.0), 1.0))
    p_base = _softmax_logits(model, x)
    if alpha < skip_threshold:
        return p_base, 0.0
    p_tta = predict_naive_tta(model, x)
    return (1.0 - alpha) * p_base + alpha * p_tta, alpha
