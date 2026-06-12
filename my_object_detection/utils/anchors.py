"""
Anchor definitions cho kiến trúc YOLOv3-style.
3 scales × 3 anchors per scale = 9 anchors tổng cộng.
Anchors được định nghĩa bằng pixel cho ảnh 640×640.
"""
import torch

# Anchors (width, height) tính bằng pixel cho ảnh 640×640
# Dựa trên YOLOv3 COCO anchors, được scale lên cho 640px
# Đã tăng kích thước anchor P5 để bắt được vật thể cực lớn (chó/mèo chiếm toàn khung hình)
ANCHORS = [
    # Scale 0: P3 (stride 8, grid 80×80) — Vật thể NHỎ
    [(31, 48), (52, 99), (70, 198)],
    # Scale 1: P4 (stride 16, grid 40×40) — Vật thể VỪA
    [(141, 138), (120, 300), (273, 259)],
    # Scale 2: P5 (stride 32, grid 20×20) — Vật thể LỚN
    [(196, 442), (347, 499), (558, 568)],
]
STRIDES = [8, 16, 32]
NUM_ANCHORS_PER_SCALE = 3

# Ngưỡng IoU tối thiểu để gán GT vào anchor (Multi-Anchor Assignment)
MULTI_ANCHOR_IOU_THRESH = 0.25


def get_anchors():
    """Trả về danh sách anchors dưới dạng list of list of tuples."""
    return ANCHORS


def get_anchor_tensors():
    """Trả về danh sách 3 tensors [3, 2], mỗi tensor chứa (w, h) cho 1 scale."""
    return [torch.tensor(scale, dtype=torch.float32) for scale in ANCHORS]


def anchor_wh_iou(anchor_wh, gt_wh):
    """
    Tính IoU dựa trên width/height (giả định centered tại gốc tọa độ).
    anchor_wh: (aw, ah) — 1 anchor
    gt_wh: (gw, gh) — 1 ground truth box
    """
    aw, ah = anchor_wh
    gw, gh = gt_wh
    inter = min(aw, gw) * min(ah, gh)
    union = aw * ah + gw * gh - inter
    return inter / union if union > 0 else 0.0


def find_best_anchor(gt_w, gt_h):
    """
    Tìm anchor phù hợp nhất cho 1 GT box dựa trên IoU width/height.
    Returns: (scale_idx, anchor_idx) — Ví dụ (1, 2) = Scale P4, Anchor thứ 3
    """
    best_iou = -1
    best_scale = 0
    best_anchor = 0

    for scale_idx, scale_anchors in enumerate(ANCHORS):
        for anchor_idx, (aw, ah) in enumerate(scale_anchors):
            iou = anchor_wh_iou((aw, ah), (gt_w, gt_h))
            if iou > best_iou:
                best_iou = iou
                best_scale = scale_idx
                best_anchor = anchor_idx

    return best_scale, best_anchor


def find_matching_anchors(gt_w, gt_h, iou_thresh=MULTI_ANCHOR_IOU_THRESH):
    """
    Tìm TẤT CẢ anchors phù hợp cho 1 GT box (Multi-Anchor Assignment).
    Trả về danh sách (iou, scale_idx, anchor_idx) đã sắp xếp giảm dần theo IoU.
    Luôn trả về ít nhất 1 anchor (anchor tốt nhất).
    """
    candidates = []
    for scale_idx, scale_anchors in enumerate(ANCHORS):
        for anchor_idx, (aw, ah) in enumerate(scale_anchors):
            iou = anchor_wh_iou((aw, ah), (gt_w, gt_h))
            candidates.append((iou, scale_idx, anchor_idx))

    # Sắp xếp theo IoU giảm dần
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates
