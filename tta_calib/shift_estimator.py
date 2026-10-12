"""Label-free test-time shift-severity estimation from BatchNorm statistic drift.

Idea: a trained model's BatchNorm running stats summarize the training
feature distribution. At test time, the per-batch feature statistics of a
shifted batch drift away from those references; the normalized drift is a
cheap, label-free proxy for shift severity. A monotonic calibrator
(isotonic regression) maps raw drift to severity in [0, 1], fitted on
validation batches with known injected severities.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression


class ShiftEstimator:
    def __init__(self, model):
        self.ref = {}
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                self.ref[name] = (
                    m.running_mean.detach().clone(),
                    m.running_var.detach().clone(),
                )
        if not self.ref:
            raise ValueError("model has no BatchNorm2d layers")
        self._hooks = []
        self._batch_stats = {}
        self.calibrator = None
        self.raw_clean = None

    def _hook_for(self, name):
        def hook(module, inp, out):
            x = inp[0].detach()
            # batch mean/var over N,H,W per channel
            bm = x.mean(dim=(0, 2, 3))
            bv = x.var(dim=(0, 2, 3), unbiased=False)
            self._batch_stats[name] = (bm, bv)

        return hook

    def attach(self, model):
        self.detach()
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                self._hooks.append(m.register_forward_hook(self._hook_for(name)))

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def raw_score(self, model, batch):
        """Normalized BN drift for one batch (model in eval mode, no grad)."""
        self._batch_stats = {}
        self.attach(model)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(batch)
        if was_training:
            model.train()
        self.detach()
        drifts = []
        for name, (rm, rv) in self.ref.items():
            bm, bv = self._batch_stats[name]
            rm = rm.to(bm.device)
            rv = rv.to(bv.device)
            d_mean = ((bm - rm) ** 2 / (rv + 1e-6)).mean()
            d_var = ((bv - rv) ** 2 / (rv + 1e-6) ** 2).mean()
            drifts.append((d_mean + d_var).item())
        return float(np.mean(drifts))

    def calibrate(self, model, val_loader, shift_fn, device="cpu", n_batches=8):
        """Fit raw->severity mapping on validation batches with known severities.

        shift_fn(batch, severity01) -> shifted batch. Uses severities
        [0.0, 0.25, 0.5, 0.75, 1.0].
        """
        raws, sevs = [], []
        severities = [0.0, 0.25, 0.5, 0.75, 1.0]
        it = iter(val_loader)
        for s in severities:
            for _ in range(n_batches):
                try:
                    x, _ = next(it)
                except StopIteration:
                    it = iter(val_loader)
                    x, _ = next(it)
                x = shift_fn(x.to(device), s)
                raws.append(self.raw_score(model, x))
                sevs.append(s)
        raws = np.array(raws)
        sevs = np.array(sevs)
        self.raw_clean = float(np.median(raws[sevs == 0.0]))
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raws, sevs)

    def severity(self, model, batch):
        """Estimated shift severity in [0, 1]."""
        raw = self.raw_score(model, batch)
        if self.calibrator is None:
            # uncalibrated fallback: saturating normalization vs clean median
            if self.raw_clean is None:
                return 0.0
            return float(np.clip((raw - self.raw_clean) / (self.raw_clean + 1e-9), 0, 1))
        return float(np.clip(self.calibrator.predict([raw])[0], 0, 1))
