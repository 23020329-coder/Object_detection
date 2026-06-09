import torch
import torch.nn as nn
import torchvision.models as models

class YoloResNet(nn.Module):
    def __init__(self, num_classes=5, S=14):
        super().__init__()
        self.S = S
        self.C = num_classes
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # PHA 3: Thay đổi từ stride=2 -> stride=1 để giữ nguyên kích thước lưới 14x14 thay vì nén xuống 7x7
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=self.C + 5, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.backbone(x)           
        x = self.head(x)               
        
        x = x.permute(0, 2, 3, 1)      
        
        out = x.clone()
        out[..., self.C:] = torch.sigmoid(x[..., self.C:])
        
        return out
