"""
YOLOv3-style Detector: ResNet50 Backbone + FPN + 3 Multi-Scale Detection Heads.
Output: 3 tensors (raw logits) cho 3 scales — KHÔNG áp dụng sigmoid/exp.
Việc activate sẽ do Loss function và Inference function thực hiện.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBnAct(nn.Module):
    """Conv2d + BatchNorm + LeakyReLU — building block cơ bản."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class YoloResNet(nn.Module):
    """
    Kiến trúc YOLOv3-style:
    - Backbone: ResNet50 (pre-trained ImageNet)
    - Neck: FPN (Feature Pyramid Network)
    - Head: 3 Detection Heads cho 3 scales (stride 8, 16, 32)

    Output per scale: [B, S, S, num_anchors, 5 + C]
        - [0]: tx logit  (sigmoid → x offset trong ô lưới)
        - [1]: ty logit  (sigmoid → y offset trong ô lưới)
        - [2]: tw logit  (exp × anchor_w → width pixel)
        - [3]: th logit  (exp × anchor_h → height pixel)
        - [4]: objectness logit (sigmoid → probability)
        - [5:5+C]: class logits (sigmoid → per-class probability)
    """

    def __init__(self, num_classes=5, num_anchors=3):
        super().__init__()
        self.C = num_classes
        self.num_anchors = num_anchors

        # === Backbone: ResNet50 ===
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(*list(resnet.children())[:4])  # conv1, bn1, relu, maxpool
        self.layer1 = resnet.layer1   # [B, 256,  H/4,  W/4]
        self.layer2 = resnet.layer2   # [B, 512,  H/8,  W/8]   → C3
        self.layer3 = resnet.layer3   # [B, 1024, H/16, W/16]  → C4
        self.layer4 = resnet.layer4   # [B, 2048, H/32, W/32]  → C5

        # === FPN: Feature Pyramid Network ===
        fpn_ch = 256

        # Lateral connections: giảm channels xuống fpn_ch
        self.lat5 = nn.Conv2d(2048, fpn_ch, 1)
        self.lat4 = nn.Conv2d(1024, fpn_ch, 1)
        self.lat3 = nn.Conv2d(512, fpn_ch, 1)

        # Smooth layers sau khi merge
        self.smooth5 = ConvBnAct(fpn_ch, fpn_ch, 3)
        self.smooth4 = ConvBnAct(fpn_ch, fpn_ch, 3)
        self.smooth3 = ConvBnAct(fpn_ch, fpn_ch, 3)

        # === 3 Detection Heads ===
        out_per_anchor = 5 + num_classes  # tx, ty, tw, th, obj, cls...
        self.head_s = self._make_head(fpn_ch, num_anchors * out_per_anchor)  # P3, small objects
        self.head_m = self._make_head(fpn_ch, num_anchors * out_per_anchor)  # P4, medium objects
        self.head_l = self._make_head(fpn_ch, num_anchors * out_per_anchor)  # P5, large objects

    def _make_head(self, in_ch, out_ch):
        """Detection Head: 3 Conv layers + 1 Prediction layer."""
        return nn.Sequential(
            ConvBnAct(in_ch, in_ch * 2, 3),          # 256 → 512
            ConvBnAct(in_ch * 2, in_ch, 1, padding=0), # 512 → 256
            ConvBnAct(in_ch, in_ch * 2, 3),          # 256 → 512
            nn.Conv2d(in_ch * 2, out_ch, 1),          # 512 → A*(5+C)
        )

    def forward(self, x):
        # === Backbone ===
        x = self.stem(x)
        c2 = self.layer1(x)    # stride 4
        c3 = self.layer2(c2)   # stride 8,  [B, 512, 80, 80]
        c4 = self.layer3(c3)   # stride 16, [B, 1024, 40, 40]
        c5 = self.layer4(c4)   # stride 32, [B, 2048, 20, 20]

        # === FPN Top-Down ===
        p5 = self.lat5(c5)                                                 # [B, 256, 20, 20]
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')  # [B, 256, 40, 40]
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')  # [B, 256, 80, 80]

        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)

        # === Detection Heads → Raw Logits ===
        out_s = self.head_s(p3)  # [B, A*(5+C), 80, 80]  — small objects
        out_m = self.head_m(p4)  # [B, A*(5+C), 40, 40]  — medium objects
        out_l = self.head_l(p5)  # [B, A*(5+C), 20, 20]  — large objects

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
