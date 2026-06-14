"""
Inference script cho YOLOv5-style Multi-Scale Detector + TTA.
Decode predictions từ 3 scales, áp dụng batched NMS, xuất predictions.json.
Hỗ trợ Test-Time Augmentation (TTA): flip ngang + merge boxes.
"""
import os
import json
import argparse
import torch
import cv2
import numpy as np
import torch
import torchvision.ops as ops
from ensemble_boxes import weighted_boxes_fusion

from model_arch import YoloResNet
from utils.metrics import decode_multi_scale
from utils.anchors import get_anchors, STRIDES


def parse_args():
    parser = argparse.ArgumentParser(description="Suy luận YOLOv5-style Detector")
    parser.add_argument("--image_dir", type=str, required=True, help="Đường dẫn đến thư mục ảnh cần dự đoán")
    parser.add_argument("--output", type=str, required=True, help="Đường dẫn file predictions.json")
    parser.add_argument("--checkpoint", type=str, default="./models/best.pth", help="Đường dẫn file mô hình .pth")
    parser.add_argument("--conf_thresh", type=float, default=0.001, help="Ngưỡng độ tin cậy")
    parser.add_argument("--iou_thresh", type=float, default=0.6, help="Ngưỡng NMS")
    parser.add_argument("--image_size", type=int, default=640, help="Kích thước ảnh")
    parser.add_argument("--tta", action="store_true", help="Bật Test-Time Augmentation (flip ngang)")
    parser.add_argument("--classes_json", type=str, default=None,
                        help="Đường dẫn JSON chứa classes (nếu không cung cấp, dùng mặc định)")
    return parser.parse_args()


def predict_image(model, image_path, device, image_size=640, threshold=0.15, iou_threshold=0.4, use_tta=False):
    """Predict trên 1 ảnh, decode multi-scale, áp dụng batched NMS. Hỗ trợ TTA."""
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

    # Decode từ 3 scales - Coi như Model 1
    boxes_list = []
    scores_list = []
    labels_list = []

    orig_boxes = decode_multi_scale(outputs, image_size, C, conf_threshold=threshold)
    if len(orig_boxes) > 0:
        b_list = [[max(0.0, min(1.0, b[0]/image_size)), max(0.0, min(1.0, b[1]/image_size)), 
                   max(0.0, min(1.0, b[2]/image_size)), max(0.0, min(1.0, b[3]/image_size))] for b in orig_boxes]
        s_list = [b[4] for b in orig_boxes]
        l_list = [int(b[5]) for b in orig_boxes]
        boxes_list.append(b_list)
        scores_list.append(s_list)
        labels_list.append(l_list)

    # === TTA: Horizontal Flip === Coi như Model 2
    if use_tta:
        img_flip = img_tensor.flip(-1)
        with torch.no_grad():
            outputs_flip = model(img_flip)
        flip_boxes = decode_multi_scale(outputs_flip, image_size, C, conf_threshold=threshold)
        if len(flip_boxes) > 0:
            b_list_flip = [[max(0.0, min(1.0, (image_size - b[2])/image_size)), max(0.0, min(1.0, b[1]/image_size)), 
                            max(0.0, min(1.0, (image_size - b[0])/image_size)), max(0.0, min(1.0, b[3]/image_size))] for b in flip_boxes]
            s_list_flip = [b[4] for b in flip_boxes]
            l_list_flip = [int(b[5]) for b in flip_boxes]
            boxes_list.append(b_list_flip)
            scores_list.append(s_list_flip)
            labels_list.append(l_list_flip)

    if len(boxes_list) == 0:
        return original_img, []

    # Sử dụng thư viện ensemble_boxes WBF
    boxes_res, scores_res, labels_res = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, 
        weights=None, iou_thr=iou_threshold, skip_box_thr=0.0
    )

    # Scale boxes về tọa độ ảnh gốc
    scaled_boxes = []
    for i in range(len(boxes_res)):
        x1, y1, x2, y2 = boxes_res[i]
        score = float(scores_res[i])
        cls_idx = int(labels_res[i])
        
        scaled_boxes.append((
            x1 * orig_w,
            y1 * orig_h,
            x2 * orig_w,
            y2 * orig_h,
            score, cls_idx
        ))

    return original_img, scaled_boxes


def generate_predictions_json(model, image_dir, output_json, classes, device,
                              image_size, threshold, iou_threshold, use_tta=False):
    """Chạy inference trên toàn bộ thư mục ảnh, xuất predictions.json."""
    predictions = []

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    for filename in image_files:
        img_path = os.path.join(image_dir, filename)
        _, final_boxes = predict_image(
            model, img_path, device, image_size, threshold, iou_threshold, use_tta=use_tta
        )

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

    tta_status = "BẬT" if args.tta else "TẮT"
    print(f"TTA (Test-Time Augmentation): {tta_status}")
    print(f"Confidence threshold: {args.conf_thresh} | NMS IoU threshold: {args.iou_thresh}")

    generate_predictions_json(
        model=model,
        image_dir=args.image_dir,
        output_json=args.output,
        classes=classes,
        device=device,
        image_size=args.image_size,
        threshold=args.conf_thresh,
        iou_threshold=args.iou_thresh,
        use_tta=args.tta
    )
