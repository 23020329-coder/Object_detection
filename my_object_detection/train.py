"""
Training script cho YOLOv5-style Detector with Decoupled Heads.
Bao gồm: AMP, Warmup, EMA, Gradient Clipping, Multi-Scale Training,
          Close Mosaic Strategy, Cosine Annealing.
"""
import math
import os
import argparse
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from utils.dataset import parse_annotations, ObjectDetectionDataset, MosaicDataset, get_train_transform
from utils.loss import YoloLoss
from utils.metrics import evaluate_model_map
from model_arch import YoloResNet


# === EMA (Exponential Moving Average) ===
class EMA:
    """Giữ bản sao 'mượt' của TOÀN BỘ model (weights + BN buffers). Dùng để evaluate & save."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.updates = 0
        # Dùng state_dict() để track TẤT CẢ: weights, biases, BN running_mean/var
        self.shadow = copy.deepcopy(model.state_dict())
        self.backup = {}

    def update(self, model):
        self.updates += 1
        # Dynamic decay cho những epoch đầu (giống YOLOv5/v8)
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        current = model.state_dict()
        for key in self.shadow:
            if current[key].dtype.is_floating_point:
                self.shadow[key] = d * self.shadow[key] + (1 - d) * current[key]
            else:
                self.shadow[key] = current[key]  # num_batches_tracked (int)

    def apply_shadow(self, model):
        self.backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self.backup)
        self.backup = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện YOLOv5-style Detector")
    parser.add_argument("--train_data", type=str, required=True, help="Đường dẫn đến file json tập train")
    parser.add_argument("--val_data", type=str, required=True, help="Đường dẫn đến file json tập validation")
    parser.add_argument("--image_dir", type=str, required=True, help="Đường dẫn đến thư mục ảnh train")
    parser.add_argument("--val_image_dir", type=str, required=True, help="Đường dẫn đến thư mục ảnh validation")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Thư mục lưu mô hình")
    parser.add_argument("--epochs", type=int, default=50, help="Số lượng epoch")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=640, help="Kích thước ảnh")
    parser.add_argument("--resume", action="store_true", help="Tự động nạp lại checkpoint nếu có")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Số epoch warmup")
    parser.add_argument("--eval_interval", type=int, default=5, help="Số epoch giữa mỗi lần đánh giá mAP")
    parser.add_argument("--conf_threshold", type=float, default=0.01, help="Validation confidence threshold")
    parser.add_argument("--nms_iou_threshold", type=float, default=0.5, help="Validation NMS IoU threshold")
    parser.add_argument("--map_iou_threshold", type=float, default=0.5, help="Validation mAP IoU threshold")
    parser.add_argument("--chair_oversample", type=float, default=1.5, help="Sampling weight for images containing chair")
    return parser.parse_args()


def build_class_oversample_weights(images_info, img_to_anns, target_class="chair", target_weight=1.5):
    weights = []
    target_weight = max(float(target_weight), 1.0)
    for img_info in images_info:
        anns = img_to_anns[img_info['id']]
        has_target = any(ann['class'] == target_class for ann in anns)
        weights.append(target_weight if has_target else 1.0)
    return torch.as_tensor(weights, dtype=torch.double)


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

    # === Dataset (ban đầu với image_size mặc định) ===
    base_train_dataset = ObjectDetectionDataset(
        img_dir=args.image_dir,
        images_info=images_info,
        img_to_anns=train_anns,
        classes=classes,
        transform=get_train_transform(image_size=args.image_size),
        image_size=args.image_size
    )

    # Mosaic ON (70%), MixUp đã bị TẮT
    train_dataset = MosaicDataset(base_train_dataset, mosaic_prob=0.7, copy_paste_prob=0.0)

    # === Khởi tạo Model, Loss, Optimizer ===
    print(f"Khởi tạo YOLOv5-ResNet50 + Decoupled Heads với ảnh {args.image_size}×{args.image_size}")
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

    # OneCycleLR kích xung lực mạnh
    total_epochs = args.epochs
    warmup_epochs = args.warmup_epochs
    sample_weights = build_class_oversample_weights(
        images_info,
        train_anns,
        target_class="chair",
        target_weight=args.chair_oversample,
    )
    train_sampler = None
    if args.chair_oversample > 1.0:
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        print(f"Chair oversampling: {args.chair_oversample:.2f}x")

    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr / 10.0, args.lr * 3],
        epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        div_factor=10.0,
        final_div_factor=100.0
    )

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

    # === Fixed-Scale & Close Mosaic Config ===
    close_mosaic_epoch = total_epochs - 15 if total_epochs > 15 else total_epochs
    print(f"Fixed-Scale Training: {args.image_size}")
    print(f"Close Mosaic Strategy: Tắt Mosaic từ epoch {close_mosaic_epoch + 1}")

    # Tạo DataLoader một lần duy nhất ngoài vòng lặp để tránh overhead fork worker
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )

    # === Training Loop ===
    print(f"Bắt đầu huấn luyện {total_epochs} epoch (warmup {warmup_epochs} epoch)...")
    for epoch in range(total_epochs):
        model.train()

        current_size = args.image_size

        # === Close Mosaic Strategy ===
        if epoch + 1 > close_mosaic_epoch:
            train_dataset.mosaic_prob = 0.0
            current_size = args.image_size  # Fix size khi fine-tune
            if epoch + 1 == close_mosaic_epoch + 1:
                print(f"\n[!] Close Mosaic: Tắt Mosaic từ epoch {epoch + 1} — fine-tune trên ảnh clean\n")
        else:
            train_dataset.mosaic_prob = 0.7

        # Cập nhật image_size cho dataset (DataLoader sẽ tự lấy kích thước mới)
        train_dataset.base.image_size = current_size
        train_dataset.base.transform = get_train_transform(image_size=current_size)
        criterion.image_size = current_size

        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{total_epochs}] (size={current_size})", leave=True)
        for images, tgt_s, tgt_m, tgt_l in loop:
            images = images.to(device)
            tgt_s = tgt_s.to(device)
            tgt_m = tgt_m.to(device)
            tgt_l = tgt_l.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out_s, out_m, out_l = model(images)
                predictions = (out_s.float(), out_m.float(), out_l.float())
                targets = (tgt_s.float(), tgt_m.float(), tgt_l.float())
                loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # OneCycleLR steps per batch
            scheduler.step()

            # EMA update
            ema.update(model)

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)

        current_lr = optimizer.param_groups[1]['lr']  # head LR
        print(f"-> Trung bình Loss Epoch {epoch + 1}: {avg_loss:.4f} | LR: {current_lr:.6f} | Size: {current_size}")

        # === Evaluation với EMA weights ===
        ema.apply_shadow(model)

        # Evaluation luôn dùng size chuẩn (640) để kết quả nhất quán
        criterion.image_size = args.image_size

        if (epoch + 1) % args.eval_interval == 0 or epoch == total_epochs - 1:
            print(f"Đang đánh giá mAP trên tập Validation (có thể mất vài phút)...")
            val_result, _ = evaluate_model_map(
                model=model,
                gt_json_path=args.val_data,
                image_dir=args.val_image_dir,
                image_size=args.image_size,
                threshold=args.conf_threshold,
                nms_iou_threshold=args.nms_iou_threshold,
                map_iou_threshold=args.map_iou_threshold
            )
            epoch_map = val_result["mAP@0.5"]
            print(f"-> mAP@{args.map_iou_threshold:.2f} Epoch {epoch + 1}: {epoch_map:.4f}")

            # Save best mAP
            if epoch_map > best_map:
                best_map = epoch_map
                torch.save(model.state_dict(), best_model_path)
                print(f"   [!] Đã lưu checkpoint tốt nhất (mAP: {best_map:.4f}) → {best_model_path}")
        else:
            print(f"-> Bỏ qua đánh giá mAP ở epoch này (đánh giá mỗi {args.eval_interval} epoch để tiết kiệm thời gian).")

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
