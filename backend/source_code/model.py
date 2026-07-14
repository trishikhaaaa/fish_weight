import torch
import torch.nn as nn
from collections import OrderedDict


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.e1 = ConvBlock(in_ch, 32)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = ConvBlock(32, 64)
        self.p2 = nn.MaxPool2d(2)
        self.bridge = ConvBlock(64, 128, dropout=0.3)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.p1(s1))
        b = self.bridge(self.p2(s2))
        x = self.d2(torch.cat([self.up2(b), s2], dim=1))
        x = self.d1(torch.cat([self.up1(x), s1], dim=1))
        return torch.sigmoid(self.out(x))


def dice_loss(pred, target, smooth=1e-6):
    p = pred.view(-1)
    t = target.view(-1)
    return 1.0 - (2.0 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)


def bce_dice_loss(pred, target, alpha=0.1, beta=0.9, smooth=1e-5):
    # ── C1: BCE + Dice ────────────────────────────────
    pred = pred.float()
    target = target.float()
    pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
    
    bce = nn.functional.binary_cross_entropy(pred, target)
    
    p = pred.view(-1)
    t = target.view(-1)
    dice = 1.0 - (2.0 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)
    
    return alpha * bce + beta * dice


def tversky_loss(pred, target, alpha=0.3, beta=0.7, smooth=1e-5):
    # ── C2: Tversky Loss ──────────────────────────────
    # alpha=0.3, beta=0.7 penalizes false negatives more (good for small regions)
    p = pred.view(-1)
    t = target.view(-1)
    
    tp = (p * t).sum()
    fp = ((1 - t) * p).sum()
    fn = (t * (1 - p)).sum()
    
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky


def focal_dice_loss(pred, target, alpha=0.8, gamma=2.0, smooth=1e-5):
    # ── C3: Focal + Dice Loss ─────────────────────────
    pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
    
    # Focal Loss component
    bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
    p_t = pred * target + (1 - pred) * (1 - target)
    
    # Proper class balancing: alpha for fish (1), (1-alpha) for background (0)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    
    focal = alpha_t * ((1 - p_t) ** gamma) * bce
    focal_loss = focal.mean()
    
    # Dice Loss component
    p = pred.view(-1)
    t = target.view(-1)
    dice = 1.0 - (2.0 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)
    
    # Focal loss values are naturally very small, so we scale it up to prevent Dice from overpowering it
    return 10.0 * focal_loss + dice

def dice_coeff(pred, target, smooth=1e-6):
    pb = (pred > 0.45).float()
    t = target
    return ((2.0 * (pb * t).sum() + smooth) / (pb.sum() + t.sum() + smooth)).item()


def iou_metric(pred, target, smooth=1e-6):
    pb = (pred > 0.45).float()
    t = target
    inter = (pb * t).sum()
    return ((inter + smooth) / (pb.sum() + t.sum() - inter + smooth)).item()


def get_segmentation_weights(net):
    """Extract segmentation model weights as a list of NumPy arrays (for Flower FL)."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_segmentation_weights(net, parameters):
    """Set segmentation model weights from a list of NumPy arrays (from Flower FL)."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def load_segmentation_model(net, path, device="cpu"):
    """Load pretrained PyTorch segmentation weights from a file."""
    state_dict = torch.load(path, map_location=device)
    # strict=False allows loading even if some layers don't perfectly match (e.g., final classification layer)
    net.load_state_dict(state_dict, strict=False)
    return net


class BPNN(nn.Module):
    def __init__(self, input_dim=5, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            self._make_block(input_dim, 64, dropout),
            self._make_block(64, 128, dropout),
            self._make_block(128, 64, dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def _make_block(self, in_f, out_f, dropout):
        return nn.Sequential(
            nn.Linear(in_f, out_f),
            nn.BatchNorm1d(out_f),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def get_bpnn_weights(net):
    """Extract BPNN model weights as a list of NumPy arrays (for Flower FL)."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_bpnn_weights(net, parameters):
    """Set BPNN model weights from a list of NumPy arrays (from Flower FL)."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def load_bpnn_model(net, path, device="cpu"):
    """Load pretrained PyTorch BPNN weights from a file."""
    state_dict = torch.load(path, map_location=device)
    net.load_state_dict(state_dict, strict=False)
    return net
