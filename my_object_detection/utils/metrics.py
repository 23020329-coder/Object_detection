import json
import os
from collections import defaultdict
import cv2
import torch
import torchvision.ops as ops

def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union

def compute_ap(recalls, precisions):
    if not recalls:
        return 0.0

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]

    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])

    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap

def evaluate_map50(ground_truth, predictions, classes, iou_threshold=0.5):
    gt_by_class = {class_name: defaultdict(list) for class_name in classes}
    for annotation in ground_truth["annotations"]:
        gt_by_class[annotation["class"]][annotation["image_id"]].append(
            {"bbox": [float(value) for value in annotation["bbox"]], "matched": False}
        )

    pred_by_class = {class_name: [] for class_name in classes}
    for prediction in predictions:
        for box in prediction["boxes"]:
            pred_by_class[box["class"]].append(
                {
                    "image_id": prediction["image_id"],
                    "class": box["class"],
                    "confidence": float(box["confidence"]),
                    "bbox": [float(value) for value in box["bbox"]],
                }
            )

    per_class = {}
    aps = []
    total_tp = 0
    total_fp = 0
    total_gt = 0

    for class_name in classes:
        class_gt = gt_by_class[class_name]
        num_gt = sum(len(items) for items in class_gt.values())
        class_preds = sorted(pred_by_class[class_name], key=lambda item: item["confidence"], reverse=True)

        tp_flags = []
        fp_flags = []

        for prediction in class_preds:
            candidates = class_gt.get(prediction["image_id"], [])
            best_iou = 0.0
            best_index = -1

            for index, gt in enumerate(candidates):
                if gt["matched"]:
                    continue
                iou = bbox_iou(prediction["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if best_index >= 0 and best_iou >= iou_threshold:
                candidates[best_index]["matched"] = True
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        cumulative_tp = []
        cumulative_fp = []
        tp_sum = 0
        fp_sum = 0
        for tp, fp in zip(tp_flags, fp_flags):
            tp_sum += tp
            fp_sum += fp
            cumulative_tp.append(tp_sum)
            cumulative_fp.append(fp_sum)

        recalls = [value / num_gt if num_gt else 0.0 for value in cumulative_tp]
        precisions = [tp / max(tp + fp, 1) for tp, fp in zip(cumulative_tp, cumulative_fp)]
        ap = compute_ap(recalls, precisions) if num_gt else 0.0

        if num_gt:
            aps.append(ap)

        total_tp += tp_sum
        total_fp += fp_sum
        total_gt += num_gt

        per_class[class_name] = {
            "ap": round(ap, 6),
            "num_ground_truth": num_gt,
            "num_predictions": len(class_preds),
            "true_positives": tp_sum,
            "false_positives": fp_sum,
            "recall": round(tp_sum / num_gt, 6) if num_gt else 0.0,
            "precision": round(tp_sum / max(tp_sum + fp_sum, 1), 6),
        }

    map_50 = sum(aps) / len(aps) if aps else 0.0
    return {
        "mAP@0.5": round(map_50, 6),
        "iou_threshold": iou_threshold,
        "num_ground_truth_boxes": total_gt,
        "num_predictions": sum(len(item["boxes"]) for item in predictions),
        "micro_precision": round(total_tp / max(total_tp + total_fp, 1), 6),
        "micro_recall": round(total_tp / total_gt, 6) if total_gt else 0.0,
        "per_class": per_class,
    }


def predict_image_for_eval(model, image_path, threshold=0.15, iou_threshold=0.4):
    device = next(model.parameters()).device
    model.eval()

    # Tự động lấy cấu hình từ mô hình ResNet
    C = getattr(model, 'C', 5)
    S = getattr(model, 'S', 7)
    image_size = S * 32

    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = original_img.shape[:2]
    img_resized = cv2.resize(original_img, (image_size, image_size))

    img_tensor = (img_resized / 255.0 - torch.tensor([0.485, 0.456, 0.406]).numpy()) / torch.tensor([0.229, 0.224, 0.225]).numpy()
    img_tensor = torch.tensor(img_tensor).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    boxes = []
    max_obj = 0.0
    for i in range(S):
        for j in range(S):
            obj_score = predictions[0, i, j, C].item()
            max_obj = max(max_obj, obj_score)

            if obj_score < threshold:
                continue

            class_probs = torch.softmax(predictions[0, i, j, :C], dim=0)
            class_idx = torch.argmax(class_probs).item()
            class_score = class_probs[class_idx].item()
            final_score = obj_score * class_score

            if final_score < threshold:
                continue

            x, y, w, h = predictions[0, i, j, C+1:C+5]
            cx = (j + x.item()) * (image_size / S)
            cy = (i + y.item()) * (image_size / S)
            bw = w.item() * image_size
            bh = h.item() * image_size

            x1 = max(0.0, cx - bw / 2)
            y1 = max(0.0, cy - bh / 2)
            x2 = min(float(image_size), cx + bw / 2)
            y2 = min(float(image_size), cy + bh / 2)

            x1 = x1 * orig_w / float(image_size)
            y1 = y1 * orig_h / float(image_size)
            x2 = x2 * orig_w / float(image_size)
            y2 = y2 * orig_h / float(image_size)

            boxes.append((x1, y1, x2, y2, final_score, class_idx))

    if len(boxes) == 0:
        return original_img, [], max_obj

    box_tensor = torch.tensor([[b[0], b[1], b[2], b[3]] for b in boxes], dtype=torch.float32)
    score_tensor = torch.tensor([b[4] for b in boxes], dtype=torch.float32)
    keep_idx = ops.nms(box_tensor, score_tensor, iou_threshold)
    final_boxes = [boxes[i] for i in keep_idx.tolist()]

    return original_img, final_boxes, max_obj

def build_predictions_for_split(model, gt_json_path, image_dir, threshold=0.15, iou_threshold=0.4, max_detections_per_image=100):
    with open(gt_json_path, "r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    classes = ground_truth["classes"]
    predictions = []

    for image_info in ground_truth["images"]:
        image_path = os.path.join(image_dir, os.path.basename(image_info["file_name"]))
        _, boxes, max_obj = predict_image_for_eval(
            model,
            image_path,
            threshold=threshold,
            iou_threshold=iou_threshold,
        )

        boxes = sorted(boxes, key=lambda item: item[4], reverse=True)[:max_detections_per_image]
        image_predictions = []
        for x1, y1, x2, y2, confidence, class_idx in boxes:
            class_idx = int(class_idx)
            image_predictions.append(
                {
                    "class": classes[class_idx],
                    "confidence": float(confidence),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

        predictions.append({"image_id": image_info["id"], "boxes": image_predictions})

    return ground_truth, predictions

def evaluate_model_map(model, gt_json_path, image_dir, threshold=0.15, iou_threshold=0.5, max_detections_per_image=100, output_path=None):
    ground_truth, predictions = build_predictions_for_split(
        model=model,
        gt_json_path=gt_json_path,
        image_dir=image_dir,
        threshold=threshold,
        iou_threshold=iou_threshold,
        max_detections_per_image=max_detections_per_image,
    )

    result = evaluate_map50(
        ground_truth=ground_truth,
        predictions=predictions,
        classes=ground_truth["classes"],
        iou_threshold=iou_threshold,
    )

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

    return result, predictions
