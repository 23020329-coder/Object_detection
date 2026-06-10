"""
Inference script cho YOLOv3-style Multi-Scale Detector.
Decode predictions từ 3 scales, áp dụng batched NMS, xuất predictions.json.
"""
import os
import json
import argparse
import torch
import cv2
import numpy as np
import torchvision.ops as ops

from model_arch import YoloResNet
from utils.metrics import decode_multi_scale
from utils.anchors import get_anchors, STRIDES


def parse_args():
    parser = argparse.ArgumentParser(description="Suy luận YOLOv3-style Detector")
    parser.add_argument("--image_dir", type=str, required=True, help="Đường dẫn đến thư mục ảnh cần dự đoán")
    parser.add_argument("--output", type=str, required=True, help="Đường dẫn file predictions.json")
    parser.add_argument("--checkpoint", type=str, default="./models/best.pth", help="Đường dẫn file mô hình .pth")
    parser.add_argument("--conf_thresh", type=float, default=0.15, help="Ngưỡng độ tin cậy")
    parser.add_argument("--iou_thresh", type=float, default=0.4, help="Ngưỡng NMS")
    parser.add_argument("--image_size", type=int, default=640, help="Kích thước ảnh")
    parser.add_argument("--classes_json", type=str, default=None,
                        help="Đường dẫn JSON chứa classes (nếu không cung cấp, dùng mặc định)")
    return parser.parse_args()


def predict_image(model, image_path, device, image_size=640, threshold=0.15, iou_threshold=0.4):
    """Predict trên 1 ảnh, decode multi-scale, áp dụng batched NMS."""
    model.eval()
    C = model.C

    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = original_img.shape[:2]
    img_resized = cv2.resize(original_img, (image_size, image_size))

    # Normalize
    img_tensor = (img_resized / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_tensor = torch.tensor(img_tensor).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    # Decode từ 3 scales — sử dụng hàm chung từ metrics.py
    boxes = decode_multi_scale(outputs, image_size, C, conf_threshold=threshold)

    if len(boxes) == 0:
        return original_img, []

    # Batched NMS (NMS riêng cho từng class)
    box_tensor = torch.tensor([[b[0], b[1], b[2], b[3]] for b in boxes], dtype=torch.float32)
    score_tensor = torch.tensor([b[4] for b in boxes], dtype=torch.float32)
    class_tensor = torch.tensor([b[5] for b in boxes], dtype=torch.int64)

    keep_idx = ops.batched_nms(box_tensor, score_tensor, class_tensor, iou_threshold)
    final_boxes = [boxes[i] for i in keep_idx.tolist()]

    # Scale boxes về tọa độ ảnh gốc
    scaled_boxes = []
    for x1, y1, x2, y2, score, cls_idx in final_boxes:
        scaled_boxes.append((
            x1 * orig_w / image_size,
            y1 * orig_h / image_size,
            x2 * orig_w / image_size,
            y2 * orig_h / image_size,
            score, cls_idx
        ))

    return original_img, scaled_boxes


def generate_predictions_json(model, image_dir, output_json, classes, device,
                              image_size, threshold, iou_threshold):
    """Chạy inference trên toàn bộ thư mục ảnh, xuất predictions.json."""
    predictions = []

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    for filename in image_files:
        img_path = os.path.join(image_dir, filename)
        _, final_boxes = predict_image(model, img_path, device, image_size, threshold, iou_threshold)

        boxes_list = []
        for x1, y1, x2, y2, conf, cls_idx in final_boxes:
            boxes_list.append({
                "class": classes[int(cls_idx)],
                "confidence": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })

        predictions.append({
            "image_id": filename,
            "boxes": boxes_list
        })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu kết quả dự đoán ({len(predictions)} ảnh) vào {output_json}")


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Đọc classes từ JSON nếu có, nếu không dùng mặc định
    if args.classes_json and os.path.exists(args.classes_json):
        with open(args.classes_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            classes = data.get('classes', ["person", "car", "dog", "cat", "chair"])
        print(f"Đọc classes từ {args.classes_json}: {classes}")
    else:
        classes = ["person", "car", "dog", "cat", "chair"]
        print(f"Sử dụng classes mặc định: {classes}")

    # Khởi tạo model với đúng số classes
    model = YoloResNet(num_classes=len(classes)).to(device)

    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Đã tải checkpoint từ {args.checkpoint}")
    else:
        print(f"Cảnh báo: Chưa tìm thấy mô hình tại {args.checkpoint}.")

    generate_predictions_json(
        model=model,
        image_dir=args.image_dir,
        output_json=args.output,
        classes=classes,
        device=device,
        image_size=args.image_size,
        threshold=args.conf_thresh,
        iou_threshold=args.iou_thresh
    )
