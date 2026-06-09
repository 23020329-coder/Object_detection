import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import parse_annotations, ObjectDetectionDataset, MosaicDataset, get_train_transform
from utils.loss import YoloLoss
from model_arch import YoloResNet

def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình YOLO")
    parser.add_argument("--train_data", type=str, required=True, help="Đường dẫn đến file json của tập train")
    parser.add_argument("--val_data", type=str, required=True, help="Đường dẫn đến file json của tập validation")
    parser.add_argument("--image_dir", type=str, required=True, help="Đường dẫn đến thư mục ảnh train")
    parser.add_argument("--val_image_dir", type=str, required=True, help="Đường dẫn đến thư mục ảnh validation")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Thư mục lưu mô hình tốt nhất")
    parser.add_argument("--epochs", type=int, default=60, help="Số lượng epoch")
    parser.add_argument("--batch_size", type=int, default=8, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=640, help="Kích thước ảnh linh hoạt")
    parser.add_argument("--resume", action="store_true", help="Tự động nạp lại checkpoint nếu có")
    return parser.parse_args()

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    if not os.path.exists(args.train_data) or not os.path.exists(args.image_dir):
        print(f"Lỗi: Không tìm thấy {args.train_data} hoặc {args.image_dir}.")
        return

    classes, images_info, train_anns = parse_annotations(args.train_data)

    base_train_dataset = ObjectDetectionDataset(
        img_dir=args.image_dir,
        images_info=images_info,
        img_to_anns=train_anns,
        classes=classes,
        transform=get_train_transform(image_size=args.image_size),
        image_size=args.image_size
    )
    
    train_dataset = MosaicDataset(base_train_dataset, mosaic_prob=0.5)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    S = args.image_size // 32
    print(f"Khởi tạo mô hình ResNet50 với kích thước ảnh {args.image_size}x{args.image_size} (Lưới {S}x{S})")

    model = YoloResNet(num_classes=len(classes)).to(device)
    criterion = YoloLoss(S=S, C=len(classes)).to(device)
    # Backbone học rất chậm để giữ đặc trưng (lr/10)
    # Head học cực nhanh để bắt nhịp (lr*5)
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': args.lr / 10.0},
        {'params': model.head.parameters(), 'lr': args.lr * 5.0}
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(args.checkpoint_dir, 'best.pth')
    
    if args.resume and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"[*] Đã khôi phục thành công trọng số từ {best_model_path}. Tiếp tục huấn luyện!")

    best_loss = float('inf')

    print("Bắt đầu huấn luyện...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=True)
        for images, targets in loop:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        
        print(f"-> Trung bình Loss Epoch {epoch+1}: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"   [!] Đã lưu checkpoint mới tốt nhất vào {best_model_path}")

if __name__ == '__main__':
    args = parse_args()
    train(args)
