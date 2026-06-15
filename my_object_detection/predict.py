import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np
import torch
import torchvision.ops as ops
from tqdm import tqdm

from model_arch import YoloResNet
from utils.metrics import decode_multi_scale


DEFAULT_CLASSES = ["person", "car", "dog", "cat", "chair"]
DEFAULT_HF_REPO_ID = "Quanganh6905/my-object-detection-model"


def parse_args():
    parser = argparse.ArgumentParser(description="Run fast YOLO-style inference")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output", type=str, required=True, help="Output predictions.json path")
    parser.add_argument("--checkpoint", type=str, default="./models/best.pth", help="Model checkpoint path")
    parser.add_argument("--hf_repo_id", type=str, default=os.environ.get("HF_MODEL_REPO", DEFAULT_HF_REPO_ID), help="Optional Hugging Face repo id, e.g. username/object-detector")
    parser.add_argument("--hf_filename", type=str, default=os.environ.get("HF_MODEL_FILE", "best.pth"), help="Checkpoint filename inside the Hugging Face repo")
    parser.add_argument("--hf_revision", type=str, default=os.environ.get("HF_MODEL_REVISION", "main"), help="Hugging Face revision/branch/tag")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""), help="Optional token for private Hugging Face repos")
    parser.add_argument("--no_auto_download", action="store_true", help="Do not download checkpoint automatically if it is missing")
    parser.add_argument("--conf_thresh", type=float, default=0.01, help="Confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.6, help="NMS/WBF IoU threshold")
    parser.add_argument("--image_size", type=int, default=640, help="Inference image size")
    parser.add_argument("--tta", action="store_true", help="Enable horizontal flip TTA")
    parser.add_argument("--wbf", action="store_true", help="Use WBF instead of fast batched NMS")
    parser.add_argument("--max_candidates", type=int, default=300, help="Max boxes before NMS/WBF per image")
    parser.add_argument("--max_detections", type=int, default=100, help="Max boxes written per image")
    parser.add_argument("--class_conf", type=str, default="", help="Class-specific conf, e.g. chair:0.015,person:0.003")
    parser.add_argument("--class_nms", type=str, default="", help="Class-specific NMS IoU, e.g. chair:0.45")
    parser.add_argument("--classes_json", type=str, default=None, help="Optional JSON file containing classes")
    return parser.parse_args()


def download_checkpoint_from_hf(repo_id, filename, revision, destination, token=""):
    if not repo_id:
        raise FileNotFoundError(
            f"Checkpoint not found: {destination}. Provide --checkpoint or set --hf_repo_id."
        )

    safe_filename = urllib.parse.quote(filename, safe="/")
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{safe_filename}"
    os.makedirs(os.path.dirname(os.path.abspath(destination)) or ".", exist_ok=True)
    temp_path = destination + ".download"

    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    print(f"Checkpoint missing. Downloading from Hugging Face: {repo_id}/{filename}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0))
            with open(temp_path, "wb") as file:
                with tqdm(total=total or None, unit="B", unit_scale=True, desc="Download checkpoint") as pbar:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        file.write(chunk)
                        pbar.update(len(chunk))
        os.replace(temp_path, destination)
    except urllib.error.HTTPError as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Could not download checkpoint from {url}: HTTP {exc.code}") from exc
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return destination


def topk_boxes(boxes, max_candidates):
    if len(boxes) > max_candidates:
        return sorted(boxes, key=lambda item: item[4], reverse=True)[:max_candidates]
    return boxes


def preprocess_image(image_path, image_size, device):
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = original_img.shape[:2]
    img_resized = cv2.resize(original_img, (image_size, image_size))

    img_tensor = (img_resized / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return original_img, img_tensor, orig_w, orig_h


def boxes_to_wbf_inputs(boxes, image_size):
    boxes_norm = [
        [
            max(0.0, min(1.0, box[0] / image_size)),
            max(0.0, min(1.0, box[1] / image_size)),
            max(0.0, min(1.0, box[2] / image_size)),
            max(0.0, min(1.0, box[3] / image_size)),
        ]
        for box in boxes
    ]
    scores = [float(box[4]) for box in boxes]
    labels = [int(box[5]) for box in boxes]
    return boxes_norm, scores, labels


def parse_class_values(spec, classes):
    values = {}
    if not spec:
        return values

    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    for chunk in spec.split(","):
        if not chunk.strip():
            continue
        name, value = chunk.split(":", 1)
        name = name.strip()
        if name not in class_to_idx:
            raise ValueError(f"Unknown class in class-specific setting: {name}")
        values[class_to_idx[name]] = float(value)
    return values


def filter_by_class_conf(boxes, default_threshold, class_conf):
    if not class_conf:
        return boxes
    return [
        box for box in boxes
        if box[4] >= class_conf.get(int(box[5]), default_threshold)
    ]


def run_fast_nms(boxes, device, iou_threshold, max_detections, class_nms=None):
    if not boxes:
        return []

    if class_nms:
        kept = []
        for cls_idx in sorted({int(box[5]) for box in boxes}):
            cls_boxes = [box for box in boxes if int(box[5]) == cls_idx]
            kept.extend(run_fast_nms(
                cls_boxes,
                device,
                class_nms.get(cls_idx, iou_threshold),
                max_detections,
                class_nms=None,
            ))
        return sorted(kept, key=lambda item: item[4], reverse=True)[:max_detections]

    box_tensor = torch.tensor([[b[0], b[1], b[2], b[3]] for b in boxes], dtype=torch.float32, device=device)
    score_tensor = torch.tensor([b[4] for b in boxes], dtype=torch.float32, device=device)
    class_tensor = torch.tensor([int(b[5]) for b in boxes], dtype=torch.int64, device=device)

    keep_idx = ops.batched_nms(box_tensor, score_tensor, class_tensor, iou_threshold)
    keep_idx = keep_idx[:max_detections].tolist()
    return [boxes[index] for index in keep_idx]


def run_wbf(boxes, image_size, iou_threshold, score_threshold, max_detections):
    from ensemble_boxes import weighted_boxes_fusion

    if not boxes:
        return []

    boxes_norm, scores, labels = boxes_to_wbf_inputs(boxes, image_size)
    boxes_res, scores_res, labels_res = weighted_boxes_fusion(
        [boxes_norm],
        [scores],
        [labels],
        weights=None,
        iou_thr=iou_threshold,
        skip_box_thr=score_threshold,
    )

    merged = [
        (
            float(x1) * image_size,
            float(y1) * image_size,
            float(x2) * image_size,
            float(y2) * image_size,
            float(score),
            int(label),
        )
        for (x1, y1, x2, y2), score, label in zip(boxes_res, scores_res, labels_res)
    ]
    return sorted(merged, key=lambda item: item[4], reverse=True)[:max_detections]


def predict_image(model, image_path, device, image_size=640, threshold=0.01, iou_threshold=0.6,
                  use_tta=False, use_wbf=False, max_candidates=300, max_detections=100,
                  class_conf=None, class_nms=None):
    model.eval()
    num_classes = model.C
    original_img, img_tensor, orig_w, orig_h = preprocess_image(image_path, image_size, device)
    decode_threshold = min([threshold] + list((class_conf or {}).values()))

    with torch.inference_mode():
        outputs = model(img_tensor)

    boxes = topk_boxes(
        decode_multi_scale(outputs, image_size, num_classes, conf_threshold=decode_threshold),
        max_candidates,
    )

    if use_tta:
        with torch.inference_mode():
            outputs_flip = model(img_tensor.flip(-1))
        flip_boxes = topk_boxes(
            decode_multi_scale(outputs_flip, image_size, num_classes, conf_threshold=decode_threshold),
            max_candidates,
        )
        boxes.extend([
            (image_size - b[2], b[1], image_size - b[0], b[3], b[4], b[5])
            for b in flip_boxes
        ])

    boxes = filter_by_class_conf(boxes, threshold, class_conf or {})
    boxes = topk_boxes(boxes, max_candidates)
    if use_wbf:
        final_boxes = run_wbf(boxes, image_size, iou_threshold, threshold, max_detections)
    else:
        final_boxes = run_fast_nms(boxes, device, iou_threshold, max_detections, class_nms=class_nms or {})

    scaled_boxes = []
    for x1, y1, x2, y2, score, cls_idx in sorted(final_boxes, key=lambda item: item[4], reverse=True)[:max_detections]:
        scaled_boxes.append((
            max(0.0, min(x1 * orig_w / image_size, orig_w - 0.001)),
            max(0.0, min(y1 * orig_h / image_size, orig_h - 0.001)),
            max(0.0, min(x2 * orig_w / image_size, orig_w - 0.001)),
            max(0.0, min(y2 * orig_h / image_size, orig_h - 0.001)),
            float(score),
            int(cls_idx),
        ))

    return original_img, scaled_boxes


def generate_predictions_json(model, image_dir, output_json, classes, device,
                              image_size, threshold, iou_threshold, use_tta=False,
                              use_wbf=False, max_candidates=300, max_detections=100,
                              class_conf=None, class_nms=None):
    image_files = sorted([
        file_name for file_name in os.listdir(image_dir)
        if file_name.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    predictions = []
    mode = "WBF" if use_wbf else "NMS"
    for filename in tqdm(image_files, desc=f"Predict {mode}", dynamic_ncols=True):
        img_path = os.path.join(image_dir, filename)
        _, final_boxes = predict_image(
            model,
            img_path,
            device,
            image_size=image_size,
            threshold=threshold,
            iou_threshold=iou_threshold,
            use_tta=use_tta,
            use_wbf=use_wbf,
            max_candidates=max_candidates,
            max_detections=max_detections,
            class_conf=class_conf,
            class_nms=class_nms,
        )

        predictions.append({
            "image_id": filename,
            "boxes": [
                {
                    "class": classes[int(cls_idx)],
                    "confidence": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }
                for x1, y1, x2, y2, conf, cls_idx in final_boxes
            ],
        })

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(predictions)} image predictions to {output_json}")


def load_classes(classes_json):
    if classes_json and os.path.exists(classes_json):
        with open(classes_json, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data.get("classes", DEFAULT_CLASSES)
        if isinstance(data, list):
            return data
    return DEFAULT_CLASSES


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(args.classes_json)
    class_conf = parse_class_values(args.class_conf, classes)
    class_nms = parse_class_values(args.class_nms, classes)
    print(f"Classes: {classes}")
    print(f"Device: {device}")

    model = YoloResNet(num_classes=len(classes)).to(device)
    if not os.path.exists(args.checkpoint):
        if args.no_auto_download:
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        download_checkpoint_from_hf(
            repo_id=args.hf_repo_id,
            filename=args.hf_filename,
            revision=args.hf_revision,
            destination=args.checkpoint,
            token=args.hf_token,
        )

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(
        f"conf={args.conf_thresh} | iou={args.iou_thresh} | "
        f"tta={args.tta} | wbf={args.wbf} | max_candidates={args.max_candidates}"
    )
    if class_conf:
        print(f"Class conf overrides: {class_conf}")
    if class_nms:
        print(f"Class NMS overrides: {class_nms}")

    generate_predictions_json(
        model=model,
        image_dir=args.image_dir,
        output_json=args.output,
        classes=classes,
        device=device,
        image_size=args.image_size,
        threshold=args.conf_thresh,
        iou_threshold=args.iou_thresh,
        use_tta=args.tta,
        use_wbf=args.wbf,
        max_candidates=args.max_candidates,
        max_detections=args.max_detections,
        class_conf=class_conf,
        class_nms=class_nms,
    )
