import os
import json
import argparse
import torch
import cv2
import numpy as np
import torchvision.ops as ops

from model_arch import YoloResNet

def parse_args():
    parser = argparse.ArgumentParser(description="Suy luận mô hình YOLO")
    parser.add_argument("--image_dir", type=str, required=True, help="Đường dẫn đến thư mục ảnh cần dự đoán")
    parser.add_argument("--output", type=str, required=True, help="Đường dẫn file predictions.json để lưu kết quả")
    parser.add_argument("--checkpoint", type=str, default="./models/best.pth", help="Đường dẫn file mô hình .pth")
    parser.add_argument("--conf_thresh", type=float, default=0.15, help="Ngưỡng độ tin cậy")
    parser.add_argument("--iou_thresh", type=float, default=0.4, help="Ngưỡng NMS")
    parser.add_argument("--image_size", type=int, default=640, help="Kích thước ảnh lúc train (ví dụ 640)")
    return parser.parse_args()

def predict_image(model, image_path, device, image_size=640, threshold=0.15, iou_threshold=0.4):
    model.eval()

    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(original_img, (image_size, image_size))

    img_tensor = (img_resized / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_tensor = torch.tensor(img_tensor).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    boxes = []
    S = predictions.shape[1] 
    
    for i in range(S):
        for j in range(S):
            obj_score = predictions[0, i, j, 5].item()
            if obj_score < threshold:
                continue

            class_probs = torch.softmax(predictions[0, i, j, :5], dim=0)
            class_idx = torch.argmax(class_probs).item()
            class_score = class_probs[class_idx].item()
            final_score = obj_score * class_score

            if final_score < threshold:
                continue

            x, y, w, h = predictions[0, i, j, 6:10]
            cx = (j + x.item()) * (image_size / S)
            cy = (i + y.item()) * (image_size / S)
            bw = w.item() * image_size
            bh = h.item() * image_size

            x1 = max(0, cx - bw / 2)
            y1 = max(0, cy - bh / 2)
            x2 = min(image_size, cx + bw / 2)
            y2 = min(image_size, cy + bh / 2)

            orig_h, orig_w = original_img.shape[:2]
            x1 = x1 * orig_w / float(image_size)
            y1 = y1 * orig_h / float(image_size)
            x2 = x2 * orig_w / float(image_size)
            y2 = y2 * orig_h / float(image_size)

            boxes.append((x1, y1, x2, y2, final_score, class_idx))

    if len(boxes) == 0:
        return original_img, []

    box_tensor = torch.tensor([[b[0], b[1], b[2], b[3]] for b in boxes], dtype=torch.float32)
    score_tensor = torch.tensor([b[4] for b in boxes], dtype=torch.float32)
    class_idx_tensor = torch.tensor([b[5] for b in boxes], dtype=torch.int64)
    
    keep_idx = ops.batched_nms(box_tensor, score_tensor, class_idx_tensor, iou_threshold)
    final_boxes = [boxes[i] for i in keep_idx.tolist()]

    return original_img, final_boxes

def generate_predictions_json(model, image_dir, output_json, classes, device, image_size, threshold, iou_threshold):
    predictions = []
    
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
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
        
    print(f"Đã lưu kết quả dự đoán vào {output_json}")

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = ["person", "car", "dog", "cat", "chair"]
    
    model = YoloResNet(num_classes=len(classes)).to(device)
    
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Đã tải checkpoint từ {args.checkpoint}")
    else:
        print(f"Cảnh báo: Chưa tìm thấy mô hình tại {args.checkpoint}. Có thể cần huấn luyện trước.")
        
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
