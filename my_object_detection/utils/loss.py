"""
YOLOv3-style Multi-Scale Loss Function.
- Objectness: BCEWithLogitsLoss (ổn định hơn BCELoss, AMP-safe)
- Classification: Sigmoid Focal Loss (nhất quán với sigmoid inference)
- Regression: CIoU Loss

Cải tiến:
- Giảm lambda_noobj từ 0.5 → 0.35 để giảm phạt oan khi dataset thiếu nhãn.
- Mô hình sẽ tự tin hơn khi đoán trúng vật thể chưa được gán nhãn trong GT.
"""
import torch
import torch.nn as nn
import torchvision.ops as ops
from utils.anchors import STRIDES, get_anchor_tensors


class YoloLoss(nn.Module):
    def __init__(self, C=5, image_size=640):
        super().__init__()
        self.C = C
        self.image_size = image_size
        self.strides = STRIDES
        self.anchors = get_anchor_tensors()  # list of 3 tensors [3, 2]

        # BCEWithLogitsLoss: nhận raw logit, tự áp dụng sigmoid bên trong
        # → Không cần sigmoid trong model → AMP-safe tự nhiên
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')

        # Hệ số cân bằng các thành phần loss
        self.lambda_coord = 5.0
        self.lambda_obj = 1.0
        self.lambda_noobj = 0.35   # Giảm từ 0.5 → 0.35: giảm phạt oan khi data thiếu nhãn
        self.lambda_cls = 1.0

        # Focal Loss parameters
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0

    def forward(self, predictions, targets):
        """
        predictions: tuple (out_s, out_m, out_l), each [B, S, S, A, 5+C] raw logits
        targets: tuple (tgt_s, tgt_m, tgt_l), each [B, S, S, A, 5+C]

        Target format per anchor:
            [0] tx, [1] ty, [2] tw, [3] th, [4] objectness, [5:5+C] one-hot class
        """
        total_box_loss = 0.0
        total_obj_loss = 0.0
        total_cls_loss = 0.0
        device = predictions[0].device

        for scale_idx, (pred, tgt) in enumerate(zip(predictions, targets)):
            anchors = self.anchors[scale_idx].to(device)  # [3, 2]
            stride = self.strides[scale_idx]

            obj_mask = tgt[..., 4] == 1.0      # [B, S, S, A] — ô có vật thể
            noobj_mask = tgt[..., 4] == 0.0    # [B, S, S, A] — ô trống

            # === 1. Objectness Loss (BCEWithLogitsLoss) ===
            obj_loss_map = self.bce_obj(pred[..., 4], tgt[..., 4])
            obj_loss = (
                self.lambda_obj * obj_loss_map[obj_mask].sum() +
                self.lambda_noobj * obj_loss_map[noobj_mask].sum()
            )
            total_obj_loss += obj_loss

            num_pos = obj_mask.sum().item()
            if num_pos == 0:
                continue

            # === 2. Box Regression Loss (CIoU) ===
            pred_boxes = self._decode_boxes(pred, anchors, stride, obj_mask)
            tgt_boxes = self._decode_target_boxes(tgt, anchors, stride, obj_mask)
            box_loss = ops.complete_box_iou_loss(pred_boxes, tgt_boxes, reduction='sum')
            total_box_loss += box_loss

            # === 3. Classification Loss (Sigmoid Focal Loss) ===
            cls_pred = pred[..., 5:5 + self.C][obj_mask]   # [N, C] raw logits
            cls_tgt = tgt[..., 5:5 + self.C][obj_mask]     # [N, C] one-hot
            cls_loss = ops.sigmoid_focal_loss(
                cls_pred, cls_tgt,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
                reduction='sum'
            )
            total_cls_loss += cls_loss

        batch_size = predictions[0].shape[0]
        total_loss = (
            self.lambda_coord * total_box_loss
            + total_obj_loss
            + self.lambda_cls * total_cls_loss
        )
        return total_loss / batch_size

    def _decode_boxes(self, pred, anchors, stride, mask):
        """Decode predicted boxes → [x1, y1, x2, y2] normalized [0, 1]."""
        indices = mask.nonzero(as_tuple=False)  # [N, 4]: (batch, i, j, anchor)
        b, gi, gj, a = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]

        tx = pred[b, gi, gj, a, 0]
        ty = pred[b, gi, gj, a, 1]
        tw = pred[b, gi, gj, a, 2]
        th = pred[b, gi, gj, a, 3]

        aw = anchors[a, 0]
        ah = anchors[a, 1]

        cx = (torch.sigmoid(tx) + gj.float()) * stride / self.image_size
        cy = (torch.sigmoid(ty) + gi.float()) * stride / self.image_size
        w = aw * torch.exp(tw.clamp(max=5.0)) / self.image_size
        h = ah * torch.exp(th.clamp(max=5.0)) / self.image_size

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _decode_target_boxes(self, tgt, anchors, stride, mask):
        """Decode target boxes → [x1, y1, x2, y2] normalized [0, 1]."""
        indices = mask.nonzero(as_tuple=False)
        b, gi, gj, a = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]

        tx = tgt[b, gi, gj, a, 0]
        ty = tgt[b, gi, gj, a, 1]
        tw = tgt[b, gi, gj, a, 2]
        th = tgt[b, gi, gj, a, 3]

        aw = anchors[a, 0]
        ah = anchors[a, 1]

        cx = (tx + gj.float()) * stride / self.image_size
        cy = (ty + gi.float()) * stride / self.image_size
        w = aw * torch.exp(tw) / self.image_size
        h = ah * torch.exp(th) / self.image_size

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        return torch.stack([x1, y1, x2, y2], dim=-1)
