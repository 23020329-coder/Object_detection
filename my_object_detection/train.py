"""
Training script cho YOLOv3-style detector.
Bao gồm: AMP, Warmup, EMA, Gradient Clipping, Multi-scale targets.
"""
import math
import os
import argparse
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import parse_annotations, ObjectDetectionDataset, MosaicDataset, get_train_transform
from utils.loss import YoloLoss
from utils.metrics import evaluate_model_map
from model_arch import YoloResNet


# === EMA (Exponential Moving Average) ===
class EMA:
    """Giữ bản sao 'mượt' của trọng số model. Dùng để evaluate & save checkpoint."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện YOLOv3-style Detector")
    parser.add_argument("--train_data", type=str, required=True, help="Đường dẫn đến file json tập train")
    parser.add_argument("--val_data", type=str, required=True, help="Đường dẫn đến file json tập validation")
    parser.add_argument("--image_dir", type=str, required=True, help="Đường dẫn đến thư mục ảnh train")
    parser.add_argument("--val_image_dir", type=str, required=True, help="Đường dẫn đến thư mục ảnh validation")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Thư mục lưu mô hình")
    parser.add_argument("--epochs", type=int, default=40, help="Số lượng epoch")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=640, help="Kích thước ảnh")
    parser.add_argument("--resume", action="store_true", help="Tự động nạp lại checkpoint nếu có")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Số epoch warmup")
    return parser.parse_args()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(args.train_data) or not os.path.exists(args.image_dir):
        print(f"Lỗi: Không tìm thấy {args.train_data} hoặc {args.image_dir}.")
        return

    classes, images_info, train_anns = parse_annotations(args.train_data)
    C = len(classes)
    print(f"Số lượng class: {C} — {classes}")

    base_train_dataset = ObjectDetectionDataset(
        img_dir=args.image_dir,
        images_info=images_info,
        img_to_anns=train_anns,
        classes=classes,
        transform=get_train_transform(image_size=args.image_size),
        image_size=args.image_size
    )

    train_dataset = MosaicDataset(base_train_dataset, mosaic_prob=0.5)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )

    # === Khởi tạo Model, Loss, Optimizer ===
    print(f"Khởi tạo YOLOv3-ResNet50 (FPN multi-scale) với ảnh {args.image_size}×{args.image_size}")
    model = YoloResNet(num_classes=C).to(device)
    criterion = YoloLoss(C=C, image_size=args.image_size).to(device)

    # Differential LR: backbone chậm, FPN+heads nhanh
    backbone_params = list(model.stem.parameters()) + list(model.layer1.parameters()) + \
                      list(model.layer2.parameters()) + list(model.layer3.parameters()) + \
                      list(model.layer4.parameters())
    head_params = [p for p in model.parameters() if id(p) not in {id(bp) for bp in backbone_params}]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr / 10.0},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    # Warmup + Cosine Annealing
    warmup_epochs = args.warmup_epochs
    total_epochs = args.epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # === Checkpoint & EMA ===
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(args.checkpoint_dir, 'best.pth')

    if args.resume and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"[*] Đã khôi phục trọng số từ {best_model_path}")

    ema = EMA(model, decay=0.9999)
    scaler = torch.amp.GradScaler('cuda')
    best_loss = float('inf')
    best_map = 0.0

    # === Training Loop ===
    print(f"Bắt đầu huấn luyện {total_epochs} epoch (warmup {warmup_epochs} epoch)...")
    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{total_epochs}]", leave=True)
        for images, tgt_s, tgt_m, tgt_l in loop:
            images = images.to(device)
            tgt_s = tgt_s.to(device)
            tgt_m = tgt_m.to(device)
            tgt_l = tgt_l.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out_s, out_m, out_l = model(images)

            # Loss ở Float32
            predictions = (out_s.float(), out_m.float(), out_l.float())
            targets = (tgt_s.float(), tgt_m.float(), tgt_l.float())
            loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA update
            ema.update(model)

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()

        current_lr = optimizer.param_groups[1]['lr']  # head LR
        print(f"-> Trung bình Loss Epoch {epoch + 1}: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # === Evaluation với EMA weights ===
        ema.apply_shadow(model)

        print("Đang đánh giá mAP trên tập Validation...")
        val_result, _ = evaluate_model_map(
            model=model,
            gt_json_path=args.val_data,
            image_dir=args.val_image_dir,
            image_size=args.image_size,
            threshold=0.15,
            iou_threshold=0.5
        )
        epoch_map = val_result["mAP@0.5"]
        print(f"-> mAP@0.5 Epoch {epoch + 1}: {epoch_map:.4f}")

        # Save best mAP
        if epoch_map > best_map:
            best_map = epoch_map
            torch.save(model.state_dict(), best_model_path)
            print(f"   [!] Đã lưu checkpoint tốt nhất (mAP: {best_map:.4f}) → {best_model_path}")

        # Save best loss
        best_loss_path = os.path.join(args.checkpoint_dir, 'best_loss.pth')
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_loss_path)
            print(f"   [!] Đã lưu checkpoint (Loss: {best_loss:.4f}) → {best_loss_path}")

        ema.restore(model)

    print(f"\n=== Hoàn tất huấn luyện! Best mAP@0.5: {best_map:.4f} ===")


if __name__ == '__main__':
    args = parse_args()
    train(args)
