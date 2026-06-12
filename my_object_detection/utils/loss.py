"""
YOLOv5-style Multi-Scale Loss Function.

Nâng cấp:
- Center decode: sigmoid(tx) * 2 - 0.5 (range [-0.5, 1.5] thay vì [0, 1])
  → Hỗ trợ multi-cell assignment (model có thể predict tâm ngoài ô hiện tại)
- IoU-aware objectness: target obj = IoU(pred, gt) thay vì 1.0 cố định
  → Model tự đánh giá chất lượng box chính xác hơn, giảm false positives
- BCEWithLogitsLoss + Sigmoid Focal Loss + CIoU (giữ nguyên)
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

        # Hệ số loss — giảm lambda_coord vì multi-cell tạo nhiều positive hơn
        self.lambda_coord = 3.5
        self.lambda_obj = 1.0
        self.lambda_noobj = 0.5
        self.lambda_cls = 0.5

        # Focal Loss
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0

    def forward(self, predictions, targets):
        """
        predictions: tuple (out_s, out_m, out_l), each [B, S, S, A, 5+C]
        targets: tuple (tgt_s, tgt_m, tgt_l), each [B, S, S, A, 5+C]
        """
        total_box_loss = 0.0
        total_obj_loss = 0.0
        total_cls_loss = 0.0
        device = predictions[0].device
        num_total_pos = 0

        for scale_idx, (pred, tgt) in enumerate(zip(predictions, targets)):
            anchors = self.anchors[scale_idx].to(device)
            stride = self.strides[scale_idx]

            obj_mask = tgt[..., 4] == 1.0
            noobj_mask = tgt[..., 4] == 0.0
            num_pos = obj_mask.sum().item()
            num_total_pos += num_pos

            # Objectness target — sẽ được cập nhật với IoU nếu có positive
            obj_target = tgt[..., 4].clone()

            if num_pos > 0:
                # === 1. Decode boxes ===
                pred_boxes = self._decode_boxes(pred, anchors, stride, obj_mask)
                tgt_boxes = self._decode_target_boxes(tgt, anchors, stride, obj_mask)

                # === 2. Box Loss (CIoU) ===
                box_loss = ops.complete_box_iou_loss(pred_boxes, tgt_boxes, reduction='sum')
                total_box_loss += box_loss

                # === 3. IoU-aware Objectness Target ===
                with torch.no_grad():
                    ious = self._pairwise_iou(pred_boxes, tgt_boxes)
                    obj_target[obj_mask] = ious.clamp(0).detach()

                # === 4. Classification Loss (Sigmoid Focal Loss) ===
                cls_pred = pred[..., 5:5 + self.C][obj_mask]
                cls_tgt = tgt[..., 5:5 + self.C][obj_mask]
                cls_loss = ops.sigmoid_focal_loss(
                    cls_pred, cls_tgt,
                    alpha=self.focal_alpha, gamma=self.focal_gamma,
                    reduction='sum'
                )
                total_cls_loss += cls_loss

            # === 5. Objectness Loss ===
            obj_loss_map = self.bce_obj(pred[..., 4], obj_target)
            obj_loss = (
                self.lambda_obj * obj_loss_map[obj_mask].sum() +
                self.lambda_noobj * obj_loss_map[noobj_mask].sum()
            )
            total_obj_loss += obj_loss

        # Normalize
        num_total_pos = max(num_total_pos, 1)
        total_loss = (
            self.lambda_coord * total_box_loss / num_total_pos
            + total_obj_loss / predictions[0].shape[0]
            + self.lambda_cls * total_cls_loss / num_total_pos
        )
        return total_loss

    def _decode_boxes(self, pred, anchors, stride, mask):
        """Decode predicted boxes → [x1, y1, x2, y2] normalized [0, 1].
        YOLOv5-style: sigmoid(tx) * 2 - 0.5 cho range [-0.5, 1.5]."""
        indices = mask.nonzero(as_tuple=False)
        b, gi, gj, a = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]

        tx = pred[b, gi, gj, a, 0]
        ty = pred[b, gi, gj, a, 1]
        tw = pred[b, gi, gj, a, 2]
        th = pred[b, gi, gj, a, 3]

        aw = anchors[a, 0]
        ah = anchors[a, 1]

        # YOLOv5 center decode: range [-0.5, 1.5] thay vì [0, 1]
        cx = (torch.sigmoid(tx) * 2.0 - 0.5 + gj.float()) * stride / self.image_size
        cy = (torch.sigmoid(ty) * 2.0 - 0.5 + gi.float()) * stride / self.image_size
        w = aw * torch.exp(tw.clamp(max=5.0)) / self.image_size
        h = ah * torch.exp(th.clamp(max=5.0)) / self.image_size

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _decode_target_boxes(self, tgt, anchors, stride, mask):
        """Decode target boxes → [x1, y1, x2, y2] normalized [0, 1].
        Target tx/ty đã là raw offset, không cần sigmoid."""
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

    @staticmethod
    def _pairwise_iou(boxes1, boxes2):
        """Tính IoU cho từng cặp box tương ứng (element-wise, KHÔNG phải matrix)."""
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        return inter / (area1 + area2 - inter + 1e-7)
