"""Baseline CNN/ViT factory (ImageNet-pretrained torchvision models) and the
proposed temporal (Bi-LSTM) head."""
import torch
import torch.nn as nn
import torchvision.models as tvm

BASELINES = {
    # key: (pretty name, constructor, weights)
    "alexnet": ("AlexNet", tvm.alexnet, "IMAGENET1K_V1"),
    "vgg16": ("VGG-16", tvm.vgg16_bn, "IMAGENET1K_V1"),
    "googlenet": ("GoogLeNet", tvm.googlenet, "IMAGENET1K_V1"),
    "resnet50": ("ResNet-50", tvm.resnet50, "IMAGENET1K_V2"),
    "densenet121": ("DenseNet-121", tvm.densenet121, "IMAGENET1K_V1"),
    "mobilenetv2": ("MobileNetV2", tvm.mobilenet_v2, "IMAGENET1K_V2"),
    "efficientnetb0": ("EfficientNet-B0", tvm.efficientnet_b0, "IMAGENET1K_V1"),
    "vitb16": ("ViT-B/16", tvm.vit_b_16, "IMAGENET1K_V1"),
}


def build(name, n_cls, pretrained=True):
    _, ctor, w = BASELINES[name]
    m = ctor(weights=w if pretrained else None)
    if name == "googlenet":
        m.aux_logits = False; m.aux1 = m.aux2 = None
        m.transform_input = True  # torchvision enables this only when downloading weights; keep it for checkpoints
    # replace classifier; return (model, list_of_head_params)
    if name in ("resnet50", "googlenet"):
        m.fc = nn.Linear(m.fc.in_features, n_cls); head = m.fc
    elif name == "densenet121":
        m.classifier = nn.Linear(m.classifier.in_features, n_cls); head = m.classifier
    elif name in ("alexnet", "vgg16"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_cls); head = m.classifier
    elif name in ("mobilenetv2", "efficientnetb0"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_cls); head = m.classifier
    elif name == "vitb16":
        m.heads.head = nn.Linear(m.heads.head.in_features, n_cls); head = m.heads
    return m, list(head.parameters())


def resnet_features(m):
    """ResNet-50 trunk -> 2048-d global-pooled spatial feature f_t."""
    return nn.Sequential(*list(m.children())[:-1], nn.Flatten())


class TemporalBiLSTM(nn.Module):
    """Stage 3: Bi-LSTM over a sliding window of k spatial feature vectors.
    y = W_y [h_fwd (+) h_bwd] + b_y, followed by attention pooling over time
    so that corrupted frames can be down-weighted."""

    def __init__(self, in_dim=2048, hid=256, n_cls=8, drop=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(drop), nn.Linear(in_dim, 512), nn.GELU())
        self.lstm = nn.LSTM(512, hid, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hid, 1)
        self.out = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * hid, n_cls))

    def forward(self, f, return_att=False):  # f: (B,k,in_dim)
        h, _ = self.lstm(self.proj(f))          # (B,k,2*hid) = [h_fwd (+) h_bwd]
        a = torch.softmax(self.att(h).squeeze(-1), dim=1)
        z = (a.unsqueeze(-1) * h).sum(1)
        y = self.out(z)
        return (y, a) if return_att else y


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def gflops(m, dev="cpu"):
    from torch.utils.flop_counter import FlopCounterMode
    m = m.to(dev).eval()
    x = torch.randn(1, 3, 224, 224, device=dev)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(x)
    return fc.get_total_flops() / 2e9  # MACs-based GFLOPs convention (FLOPs/2)
