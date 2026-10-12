"""Synthetic distribution shifts at graded severities for 28x28 MedMNIST images.

Images are float tensors in [0, 1], shape (C, 28, 28) or batched (N, C, 28, 28).
Each shift type has a severity grid; severity is normalized to [0, 1] by
position in the grid so the shift estimator can be calibrated against it.
"""
import math
import torch
import torch.nn.functional as F

SHIFT_TYPES = ["noise", "blur", "contrast", "brightness"]


def severity_grid(shift_type):
    """Return list of (param, normalized_severity) for a shift type."""
    if shift_type == "noise":  # gaussian sigma
        params = [0.0, 0.05, 0.10, 0.20, 0.30]
    elif shift_type == "blur":  # gaussian radius in px
        params = [0.0, 0.5, 1.0, 1.5, 2.0]
    elif shift_type == "contrast":  # contrast factor (1 = clean)
        params = [1.0, 0.7, 0.5, 0.3]
    elif shift_type == "brightness":  # additive delta
        params = [0.0, 0.10, 0.20, 0.30]
    else:
        raise ValueError(f"unknown shift type: {shift_type}")
    n = len(params) - 1
    return [(p, i / n) for i, p in enumerate(params)]


def _gaussian_kernel1d(radius, device, dtype):
    # radius in px; sigma = radius / 2, kernel spans +-2*sigma
    sigma = max(radius / 2.0, 0.3)
    r = max(int(math.ceil(2 * sigma)), 1)
    xs = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (xs / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(x, radius):
    if radius <= 0:
        return x
    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0)
    n, c, h, w = x.shape
    k1 = _gaussian_kernel1d(radius, x.device, x.dtype)
    k2 = k1[:, None] * k1[None, :]
    kernel = k2.expand(c, 1, k2.shape[0], k2.shape[1])
    pad = k2.shape[0] // 2
    out = F.conv2d(x, kernel, padding=pad, groups=c)
    return out.squeeze(0) if single else out


def apply_shift(x, shift_type, param):
    """Apply one shift to a (C,H,W) or (N,C,H,W) float tensor in [0,1]."""
    if shift_type == "noise":
        if param <= 0:
            return x
        return torch.clamp(x + torch.randn_like(x) * param, 0.0, 1.0)
    if shift_type == "blur":
        return torch.clamp(gaussian_blur(x, param), 0.0, 1.0)
    if shift_type == "contrast":
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return torch.clamp((x - mean) * param + mean, 0.0, 1.0)
    if shift_type == "brightness":
        return torch.clamp(x + param, 0.0, 1.0)
    raise ValueError(f"unknown shift type: {shift_type}")


def apply_shift_batch(x, shift_type, param):
    """Batched convenience wrapper (N,C,H,W)."""
    return apply_shift(x, shift_type, param)
