import torch
import torch.nn as nn
import torchvision.ops as ops

class YoloLoss(nn.Module):
    def __init__(self, S=14, C=5):
        super().__init__()
        self.S = S
        self.C = C
        
        # Chuyển Objectness sang BCELoss (vì logit đã được qua sigmoid ở model)
        self.bce = nn.BCELoss(reduction='sum')
        
        self.lambda_noobj = 0.5
        self.lambda_coord = 5.0

    def forward(self, predictions, targets):
        obj_mask = targets[..., self.C] == 1.0     
        noobj_mask = targets[..., self.C] == 0.0

        # 1. REGRESSION LOSS (Thay MSE bằng CIoU Loss siêu việt)
        box_preds = predictions[..., self.C+1 : self.C+5][obj_mask]
        box_targets = targets[..., self.C+1 : self.C+5][obj_mask]
        
        if len(box_preds) > 0:
            # Chuyển (x, y, w, h) sang (x1, y1, x2, y2)
            preds_x1 = box_preds[:, 0] - box_preds[:, 2] / 2
            preds_y1 = box_preds[:, 1] - box_preds[:, 3] / 2
            preds_x2 = box_preds[:, 0] + box_preds[:, 2] / 2
            preds_y2 = box_preds[:, 1] + box_preds[:, 3] / 2
            preds_boxes = torch.stack([preds_x1, preds_y1, preds_x2, preds_y2], dim=-1)

            targs_x1 = box_targets[:, 0] - box_targets[:, 2] / 2
            targs_y1 = box_targets[:, 1] - box_targets[:, 3] / 2
            targs_x2 = box_targets[:, 0] + box_targets[:, 2] / 2
            targs_y2 = box_targets[:, 1] + box_targets[:, 3] / 2
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
        
        # 3. CLASSIFICATION LOSS (Focal Loss)
        class_preds = predictions[..., :self.C][obj_mask]
        class_targets = targets[..., :self.C][obj_mask]
        class_loss = ops.sigmoid_focal_loss(class_preds, class_targets, alpha=0.25, gamma=2.0, reduction='sum') if len(class_preds) > 0 else torch.tensor(0.0).to(predictions.device)

        # Tổng hợp Loss (Composite Loss)
        total_loss = (
            self.lambda_coord * box_loss   
            + object_loss                  
            + self.lambda_noobj * no_object_loss 
            + class_loss                   
        )

        batch_size = predictions.shape[0]
        return total_loss / batch_size
