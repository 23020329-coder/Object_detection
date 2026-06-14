"""
Metrics & Evaluation cho YOLOv3-style Multi-Scale Detector.
Decode predictions từ 3 scales, áp dụng NMS, tính mAP@0.5.
"""
import json
import math
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
import torchvision.ops as ops
from ensemble_boxes import weighted_boxes_fusion

from utils.anchors import get_anchors, STRIDES


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


def batched_weighted_nms(boxes, scores, classes, iou_threshold=0.5):
    """
    Weighted Boxes Fusion (WBF) thay thế cho NMS.
    Gộp các box đè nhau thay vì xóa, giúp làm mượt tọa độ và triệt tiêu nhiễu.
    """
    if len(boxes) == 0:
        return []
    
    max_coordinate = boxes.max() if len(boxes) > 0 else 0
    offsets = classes.to(boxes.dtype) * (max_coordinate + torch.tensor(1.0, device=boxes.device))
    boxes_offset = boxes + offsets[:, None]
    
    sorted_idx = torch.argsort(scores, descending=True)
    boxes_offset = boxes_offset[sorted_idx]
    original_boxes = boxes[sorted_idx]
    scores = scores[sorted_idx]
    original_classes = classes[sorted_idx]
    
    clusters = []
    cluster_boxes = []
    
    for i in range(len(boxes_offset)):
        box = boxes_offset[i]
        best_iou = 0.0
        best_c_idx = -1
        
        for c_idx, c_box in enumerate(cluster_boxes):
            iou = ops.box_iou(box.unsqueeze(0), c_box.unsqueeze(0)).item()
            if iou > best_iou:
                best_iou = iou
                best_c_idx = c_idx
                
        if best_iou > iou_threshold:
            clusters[best_c_idx].append(i)
            c_indices = clusters[best_c_idx]
            c_boxes_tensor = boxes_offset[c_indices]
            c_scores_tensor = scores[c_indices].unsqueeze(1)
            new_merged = (c_boxes_tensor * c_scores_tensor).sum(dim=0) / c_scores_tensor.sum()
            cluster_boxes[best_c_idx] = new_merged
        else:
            clusters.append([i])
            cluster_boxes.append(box)
            
    final_boxes = []
    for c_indices in clusters:
        c_boxes_tensor = original_boxes[c_indices]
        c_scores_tensor = scores[c_indices].unsqueeze(1)
        merged_box = (c_boxes_tensor * c_scores_tensor).sum(dim=0) / c_scores_tensor.sum()
        merged_score = scores[c_indices].max()
        cls_idx = original_classes[c_indices[0]]
        
        final_boxes.append((
            merged_box[0].item(), merged_box[1].item(), 
            merged_box[2].item(), merged_box[3].item(), 
            merged_score.item(), cls_idx.item()
        ))
        
    return final_boxes


def decode_multi_scale(model_outputs, image_size, C, conf_threshold=0.15):
    """
    Decode raw model outputs từ 3 scales thành danh sách boxes.
    Sử dụng SIGMOID cho cả objectness và classification (nhất quán với training).

    model_outputs: tuple (out_s, out_m, out_l), each [1, S, S, A, 5+C]
    Returns: list of (x1, y1, x2, y2, score, class_idx) in pixel coords of image_size
    """
    anchors = get_anchors()
    strides = STRIDES
    all_boxes = []

    for pred, scale_anchors, stride in zip(model_outputs, anchors, strides):
        pred = pred[0]  # remove batch dim: [S, S, A, 5+C]
        S = pred.shape[0]

        # Vectorized objectness filter
        obj_scores = torch.sigmoid(pred[..., 4])  # [S, S, A]
        mask = obj_scores > conf_threshold

        if not mask.any():
            continue

        indices = mask.nonzero(as_tuple=False)  # [N, 3]: (i, j, a)
        i_idx, j_idx, a_idx = indices[:, 0], indices[:, 1], indices[:, 2]

        # Anchors cho selected detections
        anchor_wh = torch.tensor(scale_anchors, device=pred.device, dtype=torch.float32)
        aw = anchor_wh[a_idx, 0]
        ah = anchor_wh[a_idx, 1]

        # Decode boxes
        tx = pred[i_idx, j_idx, a_idx, 0]
        ty = pred[i_idx, j_idx, a_idx, 1]
        tw = pred[i_idx, j_idx, a_idx, 2]
        th = pred[i_idx, j_idx, a_idx, 3]

        # YOLOv5-style decode: center + anchor-relative wh via sigmoid^2
        cx = (torch.sigmoid(tx) * 2.0 - 0.5 + j_idx.float()) * stride
        cy = (torch.sigmoid(ty) * 2.0 - 0.5 + i_idx.float()) * stride
        bw = aw * (torch.sigmoid(tw) * 2.0) ** 2
        bh = ah * (torch.sigmoid(th) * 2.0) ** 2

        x1 = (cx - bw / 2).clamp(min=0)
        y1 = (cy - bh / 2).clamp(min=0)
        x2 = (cx + bw / 2).clamp(max=image_size)
        y2 = (cy + bh / 2).clamp(max=image_size)

        # Class scores — SIGMOID (nhất quán với focal loss training)
        obj = obj_scores[i_idx, j_idx, a_idx]
        cls_logits = pred[i_idx, j_idx, a_idx, 5:5 + C]
        cls_scores = torch.sigmoid(cls_logits)
        class_score, class_idx = cls_scores.max(dim=1)

        final_score = obj * class_score

        # Score filter
        keep = final_score > conf_threshold
        if keep.any():
            for k in keep.nonzero(as_tuple=False).squeeze(1):
                all_boxes.append((
                    x1[k].item(), y1[k].item(),
                    x2[k].item(), y2[k].item(),
                    final_score[k].item(), class_idx[k].item()
                ))

    return all_boxes


def predict_image_for_eval(model, image_path, image_size=640, threshold=0.15, iou_threshold=0.4, use_tta=False):
    """Predict trên 1 ảnh, decode multi-scale, áp dụng NMS."""
    device = next(model.parameters()).device
    model.eval()
    C = model.C

    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = original_img.shape[:2]
    img_resized = cv2.resize(original_img, (image_size, image_size))

    # Normalize giống training
    img_tensor = (img_resized / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_tensor = torch.tensor(img_tensor).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        outputs = model(img_tensor)  # (out_s, out_m, out_l)

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


def build_predictions_for_split(model, gt_json_path, image_dir, image_size=640,
                                threshold=0.15, iou_threshold=0.4, max_detections_per_image=100,
                                use_tta=False):
    with open(gt_json_path, "r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    classes = ground_truth["classes"]
    predictions = []

    for image_info in ground_truth["images"]:
        image_path = os.path.join(image_dir, os.path.basename(image_info["file_name"]))
        _, boxes = predict_image_for_eval(
            model,
            image_path,
            image_size=image_size,
            threshold=threshold,
            iou_threshold=iou_threshold,
            use_tta=use_tta,
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


def evaluate_model_map(model, gt_json_path, image_dir, image_size=640,
                       threshold=0.15, iou_threshold=0.5,
                       max_detections_per_image=100, output_path=None,
                       use_tta=False):
    ground_truth, predictions = build_predictions_for_split(
        model=model,
        gt_json_path=gt_json_path,
        image_dir=image_dir,
        image_size=image_size,
        threshold=threshold,
        iou_threshold=iou_threshold,
        max_detections_per_image=max_detections_per_image,
        use_tta=use_tta,
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
