"""
YOLOv5-style Detector with Decoupled Heads (YOLOX-inspired):
ResNet50 Backbone + SPPF + FPN/PANet + 3 Decoupled Detection Heads.

Upgrades:
- Decoupled Head: Tách Classification và Regression thành 2 nhánh Conv riêng
  → Giải quyết task conflict, tăng mAP 1-3%
- Bias initialization (RetinaNet prior): sigmoid(bias) ≈ 0.01
  → Ổn định training từ epoch đầu tiên
- Output: [B, S, S, A, 5+C] raw logits (unchanged format)
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
        self.act = nn.SiLU(inplace=True)

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


class DecoupledHead(nn.Module):
    """
    Decoupled Detection Head (YOLOX-style).

    Feature ──┬── Cls Branch: Conv3x3→Conv3x3 → Conv1x1 → [A*C]
              └── Reg Branch: Conv3x3→Conv3x3 ──┬── Conv1x1 → [A*4] (box)
                                                 └── Conv1x1 → [A*1] (obj)

    Tách cls/reg giúp mỗi nhánh tối ưu cho riêng task của mình:
    - Cls cần texture/color features → "đây là chó hay mèo?"
    - Reg cần edge/corner features → "cạnh trái ở đâu?"
    Khi dùng chung weights, 2 task xung đột → cả 2 đều suboptimal.

    Output: [B, H, W, A, 5+C] (box + obj + cls per anchor)
    """
    def __init__(self, in_ch, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Classification branch — chuyên phân loại vật thể
        self.cls_branch = nn.Sequential(
            ConvBnAct(in_ch, in_ch, 3),
            ConvBnAct(in_ch, in_ch, 3),
        )
        self.cls_pred = nn.Conv2d(in_ch, num_anchors * num_classes, 1)

        # Regression + Objectness branch — chuyên định vị box
        self.reg_branch = nn.Sequential(
            ConvBnAct(in_ch, in_ch, 3),
            ConvBnAct(in_ch, in_ch, 3),
        )
        self.box_pred = nn.Conv2d(in_ch, num_anchors * 4, 1)
        self.obj_pred = nn.Conv2d(in_ch, num_anchors * 1, 1)

        # Bias initialization (RetinaNet prior trick)
        # sigmoid(-4.6) ≈ 0.01 → model bắt đầu với confidence thấp
        # Tránh massive loss ở epoch đầu, ổn định gradient
        self.obj_pred.bias.data.fill_(-4.6)
        self.cls_pred.bias.data.fill_(-4.6)

    def forward(self, x):
        B, _, H, W = x.shape
        A = self.num_anchors
        C = self.num_classes

        # Classification branch
        cls_feat = self.cls_branch(x)
        cls_out = self.cls_pred(cls_feat)                              # [B, A*C, H, W]
        cls_out = cls_out.view(B, A, C, H, W).permute(0, 3, 4, 1, 2)  # [B, H, W, A, C]

        # Regression + Objectness branch
        reg_feat = self.reg_branch(x)
        box_out = self.box_pred(reg_feat)                              # [B, A*4, H, W]
        box_out = box_out.view(B, A, 4, H, W).permute(0, 3, 4, 1, 2)  # [B, H, W, A, 4]
        obj_out = self.obj_pred(reg_feat)                              # [B, A*1, H, W]
        obj_out = obj_out.view(B, A, 1, H, W).permute(0, 3, 4, 1, 2)  # [B, H, W, A, 1]

        # Concat: [box(4), obj(1), cls(C)] = [5+C] per anchor
        out = torch.cat([box_out, obj_out, cls_out], dim=-1).contiguous()
        return out.float()  # Float32 cho AMP safety


class YoloResNet(nn.Module):
    """
    Kiến trúc YOLOv5-style + Decoupled Heads:

    Backbone: ResNet50 (pre-trained ImageNet)
         ↓
    SPPF: Mở rộng receptive field
         ↓
    FPN (Top-Down): C5→P5→P4→P3 — Truyền semantic features xuống
         ↓
    PANet (Bottom-Up): P3→N3→N4→N5 — Truyền localization features lên
         ↓
    3 Decoupled Heads: N3(80×80), N4(40×40), N5(20×20)

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
        self.sppf = SPPF(fpn_ch)

        # === FPN Top-Down ===
        self.lat4 = nn.Conv2d(1024, fpn_ch, 1)
        self.lat3 = nn.Conv2d(512, fpn_ch, 1)

        self.fpn5 = ConvBnAct(fpn_ch, fpn_ch, 3)  # smooth P5
        self.fpn4 = ConvBnAct(fpn_ch, fpn_ch, 3)  # smooth P4
        self.fpn3 = ConvBnAct(fpn_ch, fpn_ch, 3)  # smooth P3

        # === PANet Bottom-Up ===
        self.down3to4 = ConvBnAct(fpn_ch, fpn_ch, 3, stride=2)
        self.pan4_merge = ConvBnAct(fpn_ch * 2, fpn_ch, 1, padding=0)
        self.pan4_conv = ConvBnAct(fpn_ch, fpn_ch, 3)

        self.down4to5 = ConvBnAct(fpn_ch, fpn_ch, 3, stride=2)
        self.pan5_merge = ConvBnAct(fpn_ch * 2, fpn_ch, 1, padding=0)
        self.pan5_conv = ConvBnAct(fpn_ch, fpn_ch, 3)

        # === 3 Decoupled Detection Heads ===
        self.head_s = DecoupledHead(fpn_ch, num_anchors, num_classes)
        self.head_m = DecoupledHead(fpn_ch, num_anchors, num_classes)
        self.head_l = DecoupledHead(fpn_ch, num_anchors, num_classes)

    def forward(self, x):
        # === Backbone ===
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)   # stride 8
        c4 = self.layer3(c3)   # stride 16
        c5 = self.layer4(c4)   # stride 32

        # === SPPF ===
        p5 = self.lat5(c5)
        p5 = self.sppf(p5)

        # === FPN Top-Down (cascade smooth — smooth TRƯỚC khi upsample) ===
        p5 = self.fpn5(p5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p4 = self.fpn4(p4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p3 = self.fpn3(p3)

        # === PANet Bottom-Up ===
        n3 = p3
        n4 = self.pan4_merge(torch.cat([self.down3to4(n3), p4], dim=1))
        n4 = self.pan4_conv(n4)
        n5 = self.pan5_merge(torch.cat([self.down4to5(n4), p5], dim=1))
        n5 = self.pan5_conv(n5)

        # === Decoupled Detection Heads ===
        # Mỗi head trả về [B, H, W, A, 5+C] trực tiếp (đã reshape bên trong)
        out_s = self.head_s(n3)   # [B, 80, 80, A, 5+C]
        out_m = self.head_m(n4)   # [B, 40, 40, A, 5+C]
        out_l = self.head_l(n5)   # [B, 20, 20, A, 5+C]

        return out_s, out_m, out_l
