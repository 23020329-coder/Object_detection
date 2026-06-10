import torch
import torch.nn as nn
import torchvision.ops as ops

class YoloLoss(nn.Module):
    def __init__(self, S=14, C=5):
        super().__init__()
        self.S = S
        self.C = C
        
        # Objectness: BCELoss (vì logit đã qua sigmoid ở model)
        self.bce = nn.BCELoss(reduction='sum')
        
        # Classification: CrossEntropyLoss (softmax bên trong)
        # → Nhất quán với softmax dùng trong inference/metrics.py
        self.ce = nn.CrossEntropyLoss(reduction='sum')
        
        self.lambda_noobj = 0.5
        self.lambda_coord = 5.0

    def forward(self, predictions, targets):
        obj_mask = targets[..., self.C] == 1.0     
        noobj_mask = targets[..., self.C] == 0.0

        # 1. REGRESSION LOSS (CIoU Loss)
        box_preds = predictions[..., self.C+1 : self.C+5][obj_mask]
        box_targets = targets[..., self.C+1 : self.C+5][obj_mask]
        
        if len(box_preds) > 0:
            # Lấy chỉ số batch, hàng (i), cột (j) từ obj_mask
            indices = obj_mask.nonzero(as_tuple=False)
            b, i, j = indices[:, 0], indices[:, 1], indices[:, 2]
            
            # --- XỬ LÝ PREDICTIONS ---
            # Quy đổi x_cell, y_cell về tọa độ toàn ảnh [0, 1]
            pred_global_x = (j.float() + box_preds[:, 0]) / self.S
            pred_global_y = (i.float() + box_preds[:, 1]) / self.S
            pred_w = box_preds[:, 2]
            pred_h = box_preds[:, 3]
            
            preds_x1 = pred_global_x - pred_w / 2
            preds_y1 = pred_global_y - pred_h / 2
            preds_x2 = pred_global_x + pred_w / 2
            preds_y2 = pred_global_y + pred_h / 2
            preds_boxes = torch.stack([preds_x1, preds_y1, preds_x2, preds_y2], dim=-1)

            # --- XỬ LÝ TARGETS ---
            targ_global_x = (j.float() + box_targets[:, 0]) / self.S
            targ_global_y = (i.float() + box_targets[:, 1]) / self.S
            targ_w = box_targets[:, 2]
            targ_h = box_targets[:, 3]
            
            targs_x1 = targ_global_x - targ_w / 2
            targs_y1 = targ_global_y - targ_h / 2
            targs_x2 = targ_global_x + targ_w / 2
            targs_y2 = targ_global_y + targ_h / 2
            targs_boxes = torch.stack([targs_x1, targs_y1, targs_x2, targs_y2], dim=-1)

            box_loss = ops.complete_box_iou_loss(preds_boxes, targs_boxes, reduction='sum')
        else:
            box_loss = torch.tensor(0.0).to(predictions.device)
        
        # 2. OBJECTNESS LOSS (Thay MSE bằng BCE)
        obj_preds = predictions[..., self.C][obj_mask]
        obj_targets = targets[..., self.C][obj_mask]
        object_loss = self.bce(obj_preds, obj_targets) if len(obj_preds) > 0 else torch.tensor(0.0).to(predictions.device)
        
        noobj_preds = predictions[..., self.C][noobj_mask]
        noobj_targets = targets[..., self.C][noobj_mask]
        no_object_loss = self.bce(noobj_preds, noobj_targets) if len(noobj_preds) > 0 else torch.tensor(0.0).to(predictions.device)
        
        # 3. CLASSIFICATION LOSS (CrossEntropyLoss)
        class_preds = predictions[..., :self.C][obj_mask]   # [N, C] raw logits
        class_targets = targets[..., :self.C][obj_mask]     # [N, C] one-hot
        
        if len(class_preds) > 0:
            # Chuyển one-hot thành class index [N] để dùng CrossEntropyLoss
            class_indices = class_targets.argmax(dim=-1).long()  # [N]
            # CrossEntropyLoss áp dụng log-softmax bên trong → nhất quán với inference
            class_loss = self.ce(class_preds, class_indices)
        else:
            class_loss = torch.tensor(0.0).to(predictions.device)

        # Tổng hợp Loss (Composite Loss)
        total_loss = (
            self.lambda_coord * box_loss   
            + object_loss                  
            + self.lambda_noobj * no_object_loss 
            + class_loss                   
        )

        batch_size = predictions.shape[0]
        return total_loss / batch_size
