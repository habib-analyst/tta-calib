"""Model + data plumbing for tta-calib (MedMNIST 28px, CPU-friendly)."""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATASETS = {
    # name: (medmnist flag, num_classes, in_channels)
    "pneumoniamnist": ("pneumoniamnist", 2, 1),
    "dermamnist": ("dermamnist", 7, 3),
    "bloodmnist": ("bloodmnist", 8, 3),
}


def _conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = _conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class ResNet18_28(nn.Module):
    """ResNet-18 adapted for 28x28 (CIFAR-style stem, no early maxpool)."""

    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.conv1 = _conv3x3(in_channels, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._layer(64, 64, 2, stride=1)
        self.layer2 = self._layer(64, 128, 2, stride=2)
        self.layer3 = self._layer(128, 256, 2, stride=2)
        self.layer4 = self._layer(256, 512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _layer(self, in_c, out_c, n, stride):
        layers = [BasicBlock(in_c, out_c, stride)]
        for _ in range(1, n):
            layers.append(BasicBlock(out_c, out_c, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def get_model(num_classes, in_channels):
    return ResNet18_28(in_channels=in_channels, num_classes=num_classes)


def _to_float01(arr):
    arr = np.asarray(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr


def load_split(name, split, root="~/.medmnist"):
    """Return (images NCHW float32 in [0,1], labels long)."""
    import medmnist

    flag, num_classes, in_channels = DATASETS[name]
    info = medmnist.INFO[flag]
    DataClass = getattr(medmnist, info["python_class"])
    ds = DataClass(split=split, root=os.path.expanduser(root), download=False)
    imgs, labels = ds.imgs, ds.labels
    imgs = _to_float01(imgs)
    if imgs.ndim == 3:  # NHW -> N1HW
        imgs = imgs[:, None]
    else:  # NHWC -> NCHW
        imgs = imgs.transpose(0, 3, 1, 2)
    if imgs.shape[1] == 1 and in_channels == 3:
        imgs = np.repeat(imgs, 3, axis=1)
    labels = np.asarray(labels).reshape(-1)
    if labels.ndim > 1:
        labels = labels[:, 0]
    return imgs.astype(np.float32), labels.astype(np.int64)


def make_loader(imgs, labels, batch_size=128, shuffle=False):
    ds = TensorDataset(torch.from_numpy(imgs), torch.from_numpy(labels))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
