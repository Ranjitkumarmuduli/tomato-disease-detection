"""TomatoLeafNet – our own lightweight multi-scale attention CNN (same code as notebook 07)."""
import torch, torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: learns a weight (0-1) for every channel."""
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, ch // r, 1), nn.SiLU(),
                                nn.Conv2d(ch // r, ch, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class MSBlock(nn.Module):
    """Multi-scale depthwise block: 3x3, 5x5 and dilated 3x3 branches -> 1x1 fuse -> SE -> residual."""
    def __init__(self, c_in, c_out, stride=1, multiscale=True, se=True):
        super().__init__()
        def dw(k, d=1):
            return nn.Sequential(nn.Conv2d(c_in, c_in, k, stride, padding=d * (k // 2), dilation=d, groups=c_in, bias=False),
                                 nn.BatchNorm2d(c_in), nn.SiLU())
        self.branches = nn.ModuleList([dw(3), dw(5), dw(3, d=2)] if multiscale else [dw(3)])
        self.fuse = nn.Sequential(nn.Conv2d(c_in * len(self.branches), c_out, 1, bias=False), nn.BatchNorm2d(c_out))
        self.se = SEBlock(c_out) if se else nn.Identity()
        self.skip = (nn.Identity() if (stride == 1 and c_in == c_out) else
                     nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out)))
        self.act = nn.SiLU()

    def forward(self, x):
        y = torch.cat([b(x) for b in self.branches], dim=1)
        return self.act(self.se(self.fuse(y)) + self.skip(x))


class TomatoLeafNet(nn.Module):
    def __init__(self, num_classes=10, multiscale=True, se=True, widths=(32, 64, 128, 256), depths=(1, 2, 3, 2)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.SiLU())
        layers, c = [], 32
        for w, d in zip(widths, depths):
            for i in range(d):
                layers.append(MSBlock(c, w, stride=2 if i == 0 else 1, multiscale=multiscale, se=se))
                c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(c, num_classes))

    def forward_features(self, x):
        return self.features(self.stem(x))

    def forward_head(self, f):
        return self.head(f)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))
