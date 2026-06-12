"""
AutoAnchor: K-Means clustering sinh ra 9 anchors tối ưu cho dataset riêng.
Chạy TRƯỚC KHI TRAIN trên Kaggle để sinh ra anchors mới.

Usage trên Kaggle:
    !python my_object_detection/utils/kmeans_anchor.py \
        --train_data /kaggle/input/.../train.json \
        --image_dir  /kaggle/input/.../train/images \
        --image_size 640

Output: In ra ANCHORS mới, copy-paste vào anchors.py.
"""
import json
import argparse
import os
import numpy as np


def wh_iou_matrix(boxes, centroids):
    """
    Tính IoU width/height giữa mỗi box và mỗi centroid.
    boxes: [N, 2], centroids: [K, 2]
    Returns: [N, K]
    """
    b = boxes[:, np.newaxis, :]   # [N, 1, 2]
    c = centroids[np.newaxis, :, :]  # [1, K, 2]

    inter_w = np.minimum(b[..., 0], c[..., 0])
    inter_h = np.minimum(b[..., 1], c[..., 1])
    inter = inter_w * inter_h

    area_b = b[..., 0] * b[..., 1]
    area_c = c[..., 0] * c[..., 1]
    union = area_b + area_c - inter

    return inter / (union + 1e-7)


def kmeans_anchors(boxes, k=9, max_iter=300):
    """K-Means clustering dùng IoU distance."""
    n = len(boxes)
    indices = np.random.choice(n, k, replace=False)
    centroids = boxes[indices].copy()

    for iteration in range(max_iter):
        ious = wh_iou_matrix(boxes, centroids)
        assignments = np.argmax(ious, axis=1)

        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = assignments == j
            if mask.sum() > 0:
                new_centroids[j] = np.median(boxes[mask], axis=0)  # Median ổn định hơn Mean
            else:
                new_centroids[j] = boxes[np.random.randint(n)]

        if np.allclose(centroids, new_centroids, atol=0.5):
            break
        centroids = new_centroids

    return centroids


def main():
    parser = argparse.ArgumentParser(description="K-Means AutoAnchor")
    parser.add_argument("--train_data", type=str, required=True, help="Đường dẫn train.json")
    parser.add_argument("--image_dir", type=str, default=None, help="Thư mục ảnh (để lấy kích thước gốc)")
    parser.add_argument("--image_size", type=int, default=640, help="Kích thước ảnh target")
    parser.add_argument("--n_anchors", type=int, default=9, help="Số lượng anchors")
    parser.add_argument("--n_runs", type=int, default=30, help="Số lần chạy K-Means (lấy kết quả tốt nhất)")
    args = parser.parse_args()

    with open(args.train_data, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Cache kích thước ảnh gốc
    img_dims = {}
    if args.image_dir:
        try:
            import cv2
            print("Đang đọc kích thước ảnh gốc...")
            for img_info in data['images']:
                fname = os.path.basename(img_info['file_name'])
                fpath = os.path.join(args.image_dir, fname)
                if os.path.exists(fpath):
                    img = cv2.imread(fpath)
                    if img is not None:
                        h, w = img.shape[:2]
                        img_dims[img_info['id']] = (w, h)
            print(f"   Đọc được kích thước của {len(img_dims)} ảnh.")
        except ImportError:
            print("cv2 không khả dụng, dùng kích thước mặc định.")

    # Thu thập tất cả w/h của GT boxes, scale về image_size
    wh_list = []
    for ann in data['annotations']:
        xmin, ymin, xmax, ymax = ann['bbox']
        w = xmax - xmin
        h = ymax - ymin
        if w <= 2 or h <= 2:
            continue

        img_id = ann['image_id']
        if img_id in img_dims:
            orig_w, orig_h = img_dims[img_id]
            w = w * args.image_size / orig_w
            h = h * args.image_size / orig_h

        wh_list.append([w, h])

    boxes = np.array(wh_list, dtype=np.float32)
    print(f"\n=== Thống kê GT Boxes ===")
    print(f"Tổng số boxes: {len(boxes)}")
    print(f"Width:  min={boxes[:, 0].min():.1f}, max={boxes[:, 0].max():.1f}, mean={boxes[:, 0].mean():.1f}")
    print(f"Height: min={boxes[:, 1].min():.1f}, max={boxes[:, 1].max():.1f}, mean={boxes[:, 1].mean():.1f}")

    # Chạy K-Means nhiều lần, lấy kết quả tốt nhất
    best_anchors = None
    best_avg_iou = 0

    print(f"\nĐang chạy K-Means {args.n_runs} lần (k={args.n_anchors})...")
    for run in range(args.n_runs):
        anchors = kmeans_anchors(boxes, k=args.n_anchors)
        ious = wh_iou_matrix(boxes, anchors)
        avg_iou = ious.max(axis=1).mean()

        if avg_iou > best_avg_iou:
            best_avg_iou = avg_iou
            best_anchors = anchors
            print(f"   Run {run + 1}/{args.n_runs}: avg IoU = {avg_iou:.4f} ← NEW BEST")

    # Sắp xếp theo diện tích
    areas = best_anchors[:, 0] * best_anchors[:, 1]
    sorted_idx = np.argsort(areas)
    best_anchors = best_anchors[sorted_idx]

    # Chia thành 3 nhóm
    small = best_anchors[:3]
    medium = best_anchors[3:6]
    large = best_anchors[6:]

    print(f"\n{'=' * 60}")
    print(f"  AutoAnchor Results — Average IoU: {best_avg_iou:.4f}")
    print(f"{'=' * 60}")
    print(f"\n  Copy-paste đoạn sau vào file anchors.py:\n")
    print("ANCHORS = [")
    print(f"    # Scale 0: P3 (stride 8, grid 80×80) — Vật thể NHỎ")
    print(f"    [({small[0, 0]:.0f}, {small[0, 1]:.0f}), ({small[1, 0]:.0f}, {small[1, 1]:.0f}), ({small[2, 0]:.0f}, {small[2, 1]:.0f})],")
    print(f"    # Scale 1: P4 (stride 16, grid 40×40) — Vật thể VỪA")
    print(f"    [({medium[0, 0]:.0f}, {medium[0, 1]:.0f}), ({medium[1, 0]:.0f}, {medium[1, 1]:.0f}), ({medium[2, 0]:.0f}, {medium[2, 1]:.0f})],")
    print(f"    # Scale 2: P5 (stride 32, grid 20×20) — Vật thể LỚN")
    print(f"    [({large[0, 0]:.0f}, {large[0, 1]:.0f}), ({large[1, 0]:.0f}, {large[1, 1]:.0f}), ({large[2, 0]:.0f}, {large[2, 1]:.0f})],")
    print("]")

    # So sánh với anchor hiện tại
    current_anchors = np.array([
        [15, 20], [24, 46], [50, 35],
        [46, 94], [95, 69], [90, 183],
        [178, 138], [300, 340], [520, 480],
    ], dtype=np.float32)
    current_ious = wh_iou_matrix(boxes, current_anchors)
    current_avg_iou = current_ious.max(axis=1).mean()
    improvement = (best_avg_iou - current_avg_iou) / current_avg_iou * 100

    print(f"\n  So sánh:")
    print(f"    Anchor cũ (COCO): avg IoU = {current_avg_iou:.4f}")
    print(f"    Anchor mới (K-Means): avg IoU = {best_avg_iou:.4f}")
    print(f"    Cải thiện: +{improvement:.1f}%")


if __name__ == '__main__':
    main()
