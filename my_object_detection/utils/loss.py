import torch
import torch.nn as nn
import torchvision.ops as ops

class YoloLoss(nn.Module):
    def __init__(self, S=14, C=5):
        super().__init__()
        self.S = S
        self.C = C
        
        self.mse = nn.MSELoss(reduction='sum')
        
        self.lambda_noobj = 0.5
        self.lambda_coord = 5.0

    def forward(self, predictions, targets):
        obj_mask = targets[..., self.C] == 1.0     
        noobj_mask = targets[..., self.C] == 0.0

        box_preds = predictions[..., self.C+1 : self.C+5][obj_mask]
        box_targets = targets[..., self.C+1 : self.C+5][obj_mask]
        box_loss = self.mse(box_preds, box_targets)
        
        obj_preds = predictions[..., self.C][obj_mask]
        obj_targets = targets[..., self.C][obj_mask]
        object_loss = self.mse(obj_preds, obj_targets)
        
        class_preds = predictions[..., :self.C][obj_mask]
        class_targets = targets[..., :self.C][obj_mask]
        
        # PHA 2: SỬ DỤNG FOCAL LOSS THAY CHO BCE ĐỂ GIẢI QUYẾT CLASS IMBALANCE
        class_loss = ops.sigmoid_focal_loss(class_preds, class_targets, alpha=0.25, gamma=2.0, reduction='sum')

        noobj_preds = predictions[..., self.C][noobj_mask]
        noobj_targets = targets[..., self.C][noobj_mask]
        no_object_loss = self.mse(noobj_preds, noobj_targets)

        total_loss = (
            self.lambda_coord * box_loss   
            + object_loss                  
            + self.lambda_noobj * no_object_loss 
            + class_loss                   
        )

        batch_size = predictions.shape[0]
        return total_loss / batch_size
