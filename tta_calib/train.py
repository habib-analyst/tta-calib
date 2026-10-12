"""Train a ResNet-18 on a MedMNIST-28 split (CPU).

Resumable: per-epoch checkpoint `last.pt` (model+optim+epoch+best), `--resume`
picks up where it left off, `TRAIN_DONE` marker on completion. Charter rule:
jobs >15min must be resumable and launched detached.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .model import DATASETS, get_model, load_split, make_loader, count_params


def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    flag, num_classes, in_channels = DATASETS[args.dataset]
    print(f"dataset={args.dataset} classes={num_classes} ch={in_channels}", flush=True)

    tr_imgs, tr_y = load_split(args.dataset, "train", root=args.root)
    va_imgs, va_y = load_split(args.dataset, "val", root=args.root)
    tr_loader = make_loader(tr_imgs, tr_y, batch_size=args.batch, shuffle=True)
    va_loader = make_loader(va_imgs, va_y, batch_size=512, shuffle=False)

    model = get_model(num_classes, in_channels).to(device)
    print(f"params: {count_params(model):,}", flush=True)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "last.pt")
    best_acc, start_ep, log = 0.0, 0, []
    if args.resume and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        best_acc, start_ep, log = st["best_acc"], st["epoch"] + 1, st["log"]
        print(f"resumed from epoch {st['epoch']} best_acc={best_acc:.4f}", flush=True)

    for ep in range(start_ep, args.epochs):
        t0 = time.time()
        model.train()
        tot_loss, n = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * y.numel()
            n += y.numel()
        va = accuracy(model, va_loader, device)
        dt = time.time() - t0
        print(f"ep {ep:02d} loss={tot_loss/n:.4f} val_acc={va:.4f} ({dt:.0f}s)", flush=True)
        log.append({"epoch": ep, "loss": tot_loss / n, "val_acc": va, "sec": dt})
        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
        torch.save(
            {"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
             "best_acc": best_acc, "log": log},
            ckpt,
        )
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    open(os.path.join(args.out, "TRAIN_DONE"), "w").write("done\n")
    print(f"DONE best val_acc={best_acc:.4f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default="~/.medmnist")
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
