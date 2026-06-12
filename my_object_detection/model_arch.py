"""
YOLOv5-style Detector: ResNet50 Backbone + SPPF + PANet + 3 Detection Heads.

Nâng cấp từ YOLOv3-style:
- Thêm SPPF (Spatial Pyramid Pooling Fast) sau backbone → mở rộng receptive field
- FPN + PANet (cả top-down LẪN bottom-up) → feature fusion 2 chiều
- Output raw logits — KHÔNG áp dụng sigmoid/exp
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBnAct(nn.Module):
    """Conv2d + BatchNorm + SiLU (Swish) — building block cơ bản."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)  # SiLU > LeakyReLU (YOLOv5 dùng SiLU)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling Fast (YOLOv5).
    3 lần MaxPool nối tiếp + Concat → mở rộng receptive field mà không tăng params.
    """
    def __init__(self, ch, k=5):
        super().__init__()
        mid = ch // 2
        self.cv1 = ConvBnAct(ch, mid, 1, padding=0)
        self.cv2 = ConvBnAct(mid * 4, ch, 1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class YoloResNet(nn.Module):
    """
    Kiến trúc YOLOv5-style:

    Backbone: ResNet50 (pre-trained ImageNet)
         ↓
    SPPF: Mở rộng receptive field
         ↓
    FPN (Top-Down): C5→P5→P4→P3 — Truyền semantic features xuống
         ↓
    PANet (Bottom-Up): P3→N3→N4→N5 — Truyền localization features lên
         ↓
    3 Detection Heads: N3(80×80), N4(40×40), N5(20×20)

    Output per scale: [B, S, S, num_anchors, 5 + C]
    """

    def __init__(self, num_classes=5, num_anchors=3):
        super().__init__()
        self.C = num_classes
        self.num_anchors = num_anchors

        # === Backbone: ResNet50 ===
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1   # [256, H/4, W/4]
        self.layer2 = resnet.layer2   # [512, H/8, W/8]    → C3
        self.layer3 = resnet.layer3   # [1024, H/16, W/16]  → C4
        self.layer4 = resnet.layer4   # [2048, H/32, W/32]  → C5

        fpn_ch = 256

        # === SPPF: Sau lateral C5, trước FPN merge ===
        self.lat5 = nn.Conv2d(2048, fpn_ch, 1)
        self.sppf = SPPF(fpn_ch)  # 256 → 256, nhẹ và hiệu quả

        # === FPN Top-Down ===
        self.lat4 = nn.Conv2d(1024, fpn_ch, 1)
        self.lat3 = nn.Conv2d(512, fpn_ch, 1)

        self.fpn5 = ConvBnAct(fpn_ch, fpn_ch, 3)  # smooth P5
        self.fpn4 = ConvBnAct(fpn_ch, fpn_ch, 3)  # smooth P4
        self.fpn3 = ConvBnAct(fpn_ch, fpn_ch, 3)  # smooth P3

        # === PANet Bottom-Up ===
        # N3 = P3 (giữ nguyên)
        # N4 = Conv(Concat(Downsample(N3), P4))
        # N5 = Conv(Concat(Downsample(N4), P5))
        self.down3to4 = ConvBnAct(fpn_ch, fpn_ch, 3, stride=2)    # N3 → N4 spatial
        self.pan4_merge = ConvBnAct(fpn_ch * 2, fpn_ch, 1, padding=0)  # Concat → 256
        self.pan4_conv = ConvBnAct(fpn_ch, fpn_ch, 3)             # Refine N4

        self.down4to5 = ConvBnAct(fpn_ch, fpn_ch, 3, stride=2)    # N4 → N5 spatial
        self.pan5_merge = ConvBnAct(fpn_ch * 2, fpn_ch, 1, padding=0)  # Concat → 256
        self.pan5_conv = ConvBnAct(fpn_ch, fpn_ch, 3)             # Refine N5

        # === 3 Detection Heads ===
        out_per_anchor = 5 + num_classes
        self.head_s = self._make_head(fpn_ch, num_anchors * out_per_anchor)
        self.head_m = self._make_head(fpn_ch, num_anchors * out_per_anchor)
        self.head_l = self._make_head(fpn_ch, num_anchors * out_per_anchor)

    def _make_head(self, in_ch, out_ch):
        """Detection Head: 3 Conv + 1 Prediction."""
        return nn.Sequential(
            ConvBnAct(in_ch, in_ch * 2, 3),
            ConvBnAct(in_ch * 2, in_ch, 1, padding=0),
            ConvBnAct(in_ch, in_ch * 2, 3),
            nn.Conv2d(in_ch * 2, out_ch, 1),
        )

    def forward(self, x):
        # === Backbone ===
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)   # stride 8,  [B, 512, 80, 80]
        c4 = self.layer3(c3)   # stride 16, [B, 1024, 40, 40]
        c5 = self.layer4(c4)   # stride 32, [B, 2048, 20, 20]

        # === SPPF ===
        p5 = self.lat5(c5)     # [B, 256, 20, 20]
        p5 = self.sppf(p5)     # [B, 256, 20, 20] — receptive field mở rộng

        # === FPN Top-Down (cascade smooth — smooth TRƯỚC khi upsample) ===
        p5 = self.fpn5(p5)    # Smooth P5 trước
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p4 = self.fpn4(p4)    # Smooth P4 trước khi upsample cho P3
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p3 = self.fpn3(p3)    # Smooth P3

        # === PANet Bottom-Up ===
        n3 = p3                                                      # [B, 256, 80, 80]
        n4 = self.pan4_merge(torch.cat([self.down3to4(n3), p4], dim=1))
        n4 = self.pan4_conv(n4)                                      # [B, 256, 40, 40]
        n5 = self.pan5_merge(torch.cat([self.down4to5(n4), p5], dim=1))
        n5 = self.pan5_conv(n5)                                      # [B, 256, 20, 20]

        # === Detection Heads → Raw Logits ===
        out_s = self.head_s(n3)   # [B, A*(5+C), 80, 80]
        out_m = self.head_m(n4)   # [B, A*(5+C), 40, 40]
        out_l = self.head_l(n5)   # [B, A*(5+C), 20, 20]

        # Reshape: [B, A*(5+C), S, S] → [B, S, S, A, 5+C]
        out_s = self._reshape(out_s)
        out_m = self._reshape(out_m)
        out_l = self._reshape(out_l)

        return out_s, out_m, out_l

    def _reshape(self, x):
        B, _, H, W = x.shape
        x = x.view(B, self.num_anchors, 5 + self.C, H, W)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return x.float()  # Float32 cho AMP safety
