import torch
import torch.nn as nn
import torchvision.models as models

class YoloResNet(nn.Module):
    def __init__(self, num_classes=5, S=14):
        super().__init__()
        self.S = S
        self.C = num_classes
        
        # SỬA LỖI TẠI ĐÂY: Nâng cấp trực tiếp lên ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # ResNet50 xuất ra 2048 kênh đặc trưng
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Cập nhật in_channels = 2048
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=self.C + 5, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.backbone(x)           
        x = self.head(x)               
        
        x = x.permute(0, 2, 3, 1)      
        
        # [QUAN TRỌNG] Ép kiểu về Float32 trước khi tính Sigmoid để tránh tràn số (Precision Loss) do AMP
        x = x.float()
        
        out = x.clone()
        # Áp dụng sigmoid cho khoảng Objectness và Bounding Box
        out[..., self.C:] = torch.sigmoid(x[..., self.C:])
        
        return out
