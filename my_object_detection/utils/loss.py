"""
YOLOv5-style Multi-Scale Loss Function — Fixed ALL bugs.

Fixes applied:
1. [CRITICAL] tw/th: target stores raw pixel w/h, pred decoded via sigmoid²
   → CIoU compares boxes in same coordinate space
2. [HIGH] obj_loss: .mean() for BOTH pos and neg → balanced gradient
3. [MEDIUM] IoU-aware obj: pred_boxes.detach() → no gradient leakage
4. [LOW] Normalization: box+cls by num_pos, obj accumulated via mean (no extra /batch)
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
        self.anchors = get_anchor_tensors()

        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')

        # Hệ số loss — tuned cho .mean() normalization
        self.lambda_coord = 5.0
        self.lambda_obj   = 4.0   # Tăng vì .mean() cho ít positive cells
        self.lambda_noobj = 1.0
        self.lambda_cls   = 1.0

        # Focal Loss
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0

    def forward(self, predictions, targets):
        """
        predictions: tuple (out_s, out_m, out_l), each [B, S, S, A, 5+C]
        targets:     tuple (tgt_s, tgt_m, tgt_l), each [B, S, S, A, 5+C]
        """
        device = predictions[0].device
        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        num_total_pos  = 0

        for scale_idx, (pred, tgt) in enumerate(zip(predictions, targets)):
            anchors = self.anchors[scale_idx].to(device)
            stride  = self.strides[scale_idx]

            obj_mask   = tgt[..., 4] == 1.0
            noobj_mask = tgt[..., 4] == 0.0

            num_pos = obj_mask.sum().item()
            num_total_pos += num_pos

            # === Objectness target (IoU-aware nếu có positives) ===
            obj_target = tgt[..., 4].clone()

            if num_pos > 0:
                # --- Box Regression (CIoU) ---
                pred_boxes = self._decode_pred(pred, anchors, stride, obj_mask)
                tgt_boxes  = self._decode_tgt(tgt, stride, obj_mask)

                box_loss = ops.complete_box_iou_loss(pred_boxes, tgt_boxes, reduction='sum')
                total_box_loss = total_box_loss + box_loss

                # --- IoU-aware Objectness target ---
                # .detach() cắt gradient: IoU dùng như hằng số mục tiêu
                with torch.no_grad():
                    ious = self._pairwise_iou(pred_boxes.detach(), tgt_boxes)
                    obj_target[obj_mask] = ious.clamp(0, 1)

                # --- Classification (Sigmoid Focal Loss) ---
                cls_pred = pred[..., 5:5 + self.C][obj_mask]
                cls_tgt  = tgt[..., 5:5 + self.C][obj_mask]
                cls_loss = ops.sigmoid_focal_loss(
                    cls_pred, cls_tgt,
                    alpha=self.focal_alpha, gamma=self.focal_gamma,
                    reduction='sum'
                )
                total_cls_loss = total_cls_loss + cls_loss

            # --- Objectness Loss ---
            # .mean() cho CẢ pos lẫn neg → cân bằng gradient giữa 3 scales
            # P3(19200 cells), P4(4800), P5(1200) đều đóng góp BẰNG NHAU
            obj_loss_map = self.bce_obj(pred[..., 4], obj_target)

            if obj_mask.any():
                pos_obj = obj_loss_map[obj_mask].mean()
            else:
                pos_obj = torch.tensor(0.0, device=device)
            neg_obj = obj_loss_map[noobj_mask].mean()

            obj_loss = self.lambda_obj * pos_obj + self.lambda_noobj * neg_obj
            total_obj_loss = total_obj_loss + obj_loss

        # === Final Loss ===
        # box + cls: normalize by num_pos (per-positive average)
        # obj: already per-sample from .mean(), no extra normalization needed
        num_total_pos = max(num_total_pos, 1)
        total_loss = (
            self.lambda_coord * total_box_loss / num_total_pos
            + total_obj_loss
            + self.lambda_cls * total_cls_loss / num_total_pos
        )
        return total_loss

    def _decode_pred(self, pred, anchors, stride, mask):
        """
        Decode predicted logits → [x1, y1, x2, y2] normalized [0, 1].
        YOLOv5-style:
          cx = (sigmoid(tx)*2 - 0.5 + j) * stride / img_size
          w  = aw * (sigmoid(tw)*2)²  / img_size
        """
        indices = mask.nonzero(as_tuple=False)
        b, gi, gj, a = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]

        tx = pred[b, gi, gj, a, 0]
        ty = pred[b, gi, gj, a, 1]
        tw = pred[b, gi, gj, a, 2]
        th = pred[b, gi, gj, a, 3]

        aw = anchors[a, 0]
        ah = anchors[a, 1]

        cx = (torch.sigmoid(tx) * 2.0 - 0.5 + gj.float()) * stride / self.image_size
        cy = (torch.sigmoid(ty) * 2.0 - 0.5 + gi.float()) * stride / self.image_size
        w  = aw * (torch.sigmoid(tw) * 2.0) ** 2 / self.image_size
        h  = ah * (torch.sigmoid(th) * 2.0) ** 2 / self.image_size

        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

    def _decode_tgt(self, tgt, stride, mask):
        """
        Decode target boxes → [x1, y1, x2, y2] normalized [0, 1].

        Target format (từ dataset.py):
          tgt[..., 0] = tx = gx - cj  (raw grid offset)
          tgt[..., 1] = ty = gy - ci
          tgt[..., 2] = w  (raw pixel width tại image_size)
          tgt[..., 3] = h  (raw pixel height tại image_size)

        Decode:
          cx = (tx + cj) * stride / img_size = gx * stride / img_size
          w  = tw / img_size  (đã là pixel, chỉ cần normalize)
        """
        indices = mask.nonzero(as_tuple=False)
        b, gi, gj, a = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]

        tx = tgt[b, gi, gj, a, 0]
        ty = tgt[b, gi, gj, a, 1]
        tw = tgt[b, gi, gj, a, 2]   # raw w pixels
        th = tgt[b, gi, gj, a, 3]   # raw h pixels

        cx = (tx + gj.float()) * stride / self.image_size
        cy = (ty + gi.float()) * stride / self.image_size
        w  = tw / self.image_size
        h  = th / self.image_size

        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

    @staticmethod
    def _pairwise_iou(boxes1, boxes2):
        """Element-wise IoU (không phải matrix)."""
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

        inter  = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area1  = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2  = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        return inter / (area1 + area2 - inter + 1e-7)
