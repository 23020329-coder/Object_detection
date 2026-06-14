"""
Dataset cho kiến trúc YOLOv5-style Multi-Scale Anchor-Based.

Nâng cấp:
- Multi-Cell Assignment (YOLOv5-style): Mỗi GT box gán vào center cell + 2 neighbor cells
  → Tăng 3× positive samples → model học nhanh và chính xác hơn
- Mosaic 70% (tắt hoàn toàn ở 15 epochs cuối — Close Mosaic Strategy)
- MixUp đã bị TẮT vì gây ảnh "ma" khiến model confused về biên vật thể
- Target offset tx/ty: raw offset, có thể nằm ngoài [0, 1] cho neighbor cells
- Multi-Scale Training: hỗ trợ thay đổi image_size giữa các epoch
"""
import json
import math
import os
import random
from collections import defaultdict

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from utils.anchors import (
    ANCHORS, STRIDES, NUM_ANCHORS_PER_SCALE,
    find_anchor_candidates, find_matching_anchors,
    IGNORE_ANCHOR_IOU_THRESH, MULTI_ANCHOR_IOU_THRESH
)


def parse_annotations(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append({
            'class': ann['class'],
            'bbox': ann['bbox']
        })
    return data['classes'], data['images'], img_to_anns


class ObjectDetectionDataset(Dataset):
    def __init__(self, img_dir, images_info, img_to_anns, classes, transform=None, image_size=640):
        self.img_dir = img_dir
        self.images_info = images_info
        self.img_to_anns = img_to_anns
        self.classes = classes
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.transform = transform
        self.image_size = image_size
        self.C = len(classes)
        self.anchors = ANCHORS
        self.strides = STRIDES
        self.num_anchors = NUM_ANCHORS_PER_SCALE

    def __len__(self):
        return len(self.images_info)

    def load_raw(self, idx):
        img_info = self.images_info[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'].split('/')[-1])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anns = self.img_to_anns[img_info['id']]
        bboxes = [ann['bbox'] for ann in anns]
        labels = [self.class_to_idx[ann['class']] for ann in anns]

        return image, bboxes, labels

    def create_label_matrix(self, image, bboxes, labels):
        """
        Multi-Anchor + Multi-Cell Target Assignment (YOLOv5-style).

        Cho mỗi GT box:
        1. Tìm TẤT CẢ anchors phù hợp (IoU > threshold)
        2. Cho mỗi anchor, gán vào center cell + 2 neighbor cells gần nhất
        → Tăng ~3× positive samples so với single-cell assignment

        Target tx/ty: raw offset từ ô lưới, CÓ THỂ nằm ngoài [0, 1] cho neighbor cells.
        Model decode: sigmoid(pred_tx) * 2 - 0.5, range [-0.5, 1.5] → khớp với target.
        """
        targets = []
        for scale_idx in range(3):
            S = self.image_size // self.strides[scale_idx]
            target = torch.zeros((S, S, self.num_anchors, 5 + self.C))
            targets.append(target)

        for bbox, class_label in zip(bboxes, labels):
            xmin, ymin, xmax, ymax = bbox
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            w = xmax - xmin
            h = ymax - ymin

            if w <= 0 or h <= 0:
                continue

            # Bật lại Multi-Anchor Assignment (YOLOv5-style)
            # Không dùng `if idx_c > 0: break` nữa để tăng 3x lượng positive samples
            all_candidates = find_anchor_candidates(w, h)
            candidates = find_matching_anchors(w, h, MULTI_ANCHOR_IOU_THRESH)
            positive_keys = {(scale_idx, anchor_idx) for _, scale_idx, anchor_idx in candidates}

            for iou_val, scale_idx, anchor_idx in all_candidates:
                if iou_val < IGNORE_ANCHOR_IOU_THRESH or (scale_idx, anchor_idx) in positive_keys:
                    continue

                stride = self.strides[scale_idx]
                S = self.image_size // stride
                gj = min(int(cx / stride), S - 1)
                gi = min(int(cy / stride), S - 1)
                if targets[scale_idx][gi, gj, anchor_idx, 4] == 0:
                    targets[scale_idx][gi, gj, anchor_idx, 4] = -1.0

            for idx_c, (iou_val, scale_idx, anchor_idx) in enumerate(candidates):

                stride = self.strides[scale_idx]
                S = self.image_size // stride
                anchor_w, anchor_h = self.anchors[scale_idx][anchor_idx]

                # Grid coordinates (float)
                gx = cx / stride
                gy = cy / stride
                gj = min(int(gx), S - 1)
                gi = min(int(gy), S - 1)

                # Fractional offsets trong center cell
                fx = gx - gj
                fy = gy - gi

                # === Multi-Cell: center + 2 neighbors ===
                cells = [(gi, gj)]

                # Neighbor X: gần biên trái hay phải?
                if fx < 0.5 and gj > 0:
                    cells.append((gi, gj - 1))
                elif fx >= 0.5 and gj < S - 1:
                    cells.append((gi, gj + 1))

                # Neighbor Y: gần biên trên hay dưới?
                if fy < 0.5 and gi > 0:
                    cells.append((gi - 1, gj))
                elif fy >= 0.5 and gi < S - 1:
                    cells.append((gi + 1, gj))

                for ci, cj in cells:
                    if targets[scale_idx][ci, cj, anchor_idx, 4] != 1:
                        # Raw offset từ cell hiện tại (có thể < 0 hoặc > 1 cho neighbors)
                        tx = gx - cj
                        ty = gy - ci
                        # Lưu w, h pixel trực tiếp (KHÔNG encode log/exp)
                        # Loss sẽ decode pred bằng sigmoid², so sánh trực tiếp với w/h pixel
                        tw = w
                        th = h

                        targets[scale_idx][ci, cj, anchor_idx, 0] = tx
                        targets[scale_idx][ci, cj, anchor_idx, 1] = ty
                        targets[scale_idx][ci, cj, anchor_idx, 2] = tw
                        targets[scale_idx][ci, cj, anchor_idx, 3] = th
                        targets[scale_idx][ci, cj, anchor_idx, 4] = 1.0
                        targets[scale_idx][ci, cj, anchor_idx, 5 + int(class_label)] = 1.0

        return image, targets[0], targets[1], targets[2]

    def __getitem__(self, idx):
        image, bboxes, labels = self.load_raw(idx)

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['class_labels']

        return self.create_label_matrix(image, bboxes, labels)


class MosaicDataset(Dataset):
    """
    Dataset Wrapper: Mosaic + Normal (MixUp đã bị TẮT).

    - Mosaic: Ghép 4 ảnh thành 1 → tăng context, đa dạng hóa
    - Normal: Giữ ảnh gốc clean → tránh domain gap

    Close Mosaic Strategy (YOLOv8):
    - Epoch 1-35: Mosaic ON (70%)
    - Epoch 36-50: Mosaic OFF (0%) → fine-tune trên ảnh clean
    """

    def __init__(self, base_dataset, mosaic_prob=0.7, copy_paste_prob=0.0):
        self.base = base_dataset
        self.mosaic_prob = mosaic_prob
        self.copy_paste_prob = copy_paste_prob
        # Index riêng cho chair (class index 4)
        self._build_chair_index()

    def __len__(self):
        return len(self.base)

    def _build_chair_index(self):
        """Cache danh sách ảnh có chứa chair để sample nhanh."""
        self.chair_indices = []
        chair_idx = self.base.class_to_idx.get('chair', 4)
        for i, img_info in enumerate(self.base.images_info):
            anns = self.base.img_to_anns[img_info['id']]
            if any(self.base.class_to_idx[a['class']] == chair_idx for a in anns):
                self.chair_indices.append(i)

    def __getitem__(self, idx):
        if random.random() < self.mosaic_prob:
            result = self._mosaic(idx)
        else:
            result = self.base[idx]

        # Copy-paste chair vào ảnh hiện tại
        if self.copy_paste_prob > 0 and random.random() < self.copy_paste_prob and self.chair_indices:
            result = self._copy_paste_chair(result)

        return result

    def _copy_paste_chair(self, original_result):
        """Lấy chair crops từ ảnh khác, paste vào ảnh hiện tại."""
        img_tensor, tgt_s, tgt_m, tgt_l = original_result
        
        # Chuyển tensor về numpy để xử lý
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_np = ((img_tensor * std + mean) * 255).permute(1,2,0).byte().numpy()
        
        sz = self.base.image_size
        new_bboxes, new_labels = [], []
        
        src_idx = random.choice(self.chair_indices)
        src_img, src_bboxes, src_labels = self.base.load_raw(src_idx)
        src_img = cv2.resize(src_img, (sz, sz))
        src_h, src_w = src_img.shape[:2]
        
        chair_class_idx = self.base.class_to_idx.get('chair', 4)
        
        for bbox, label in zip(src_bboxes, src_labels):
            if label != chair_class_idx:
                continue
            
            xmin, ymin, xmax, ymax = [int(v) for v in bbox]
            xmin = max(0, int(xmin / src_w * sz))
            ymin = max(0, int(ymin / src_h * sz))
            xmax = min(sz, int(xmax / src_w * sz))
            ymax = min(sz, int(ymax / src_h * sz))
            
            if xmax - xmin < 16 or ymax - ymin < 16:
                continue
            
            # Chọn vị trí paste ngẫu nhiên
            pw = xmax - xmin
            ph = ymax - ymin
            px = random.randint(0, max(0, sz - pw))
            py = random.randint(0, max(0, sz - ph))
            
            # Paste với alpha blend nhẹ để tự nhiên hơn
            crop = src_img[ymin:ymax, xmin:xmax]
            alpha = random.uniform(0.7, 1.0)
            roi = img_np[py:py+ph, px:px+pw].astype(float)
            img_np[py:py+ph, px:px+pw] = (alpha * crop + (1-alpha) * roi).astype(np.uint8)
            
            new_bboxes.append([px, py, px+pw, py+ph])
            new_labels.append(chair_class_idx)
        
        if not new_bboxes:
            return original_result
        
        # Re-normalize và rebuild targets
        img_new = (img_np / 255.0 - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
        img_tensor_new = torch.tensor(img_new).permute(2,0,1).float()
        
        return self.base.create_label_matrix(img_tensor_new, new_bboxes, new_labels)

    def _mosaic(self, idx):
        """Mosaic: Ghép 4 ảnh thành 1 (YOLOv4-style)."""
        indices = [idx] + [random.randint(0, len(self.base) - 1) for _ in range(3)]

        sz = self.base.image_size
        mosaic_image = np.zeros((sz * 2, sz * 2, 3), dtype=np.uint8)
        mosaic_bboxes = []
        mosaic_labels = []

        for i, index in enumerate(indices):
            image, bboxes, labels = self.base.load_raw(index)
            h, w = image.shape[:2]

            image = cv2.resize(image, (sz, sz))

            x_offset = sz if i % 2 == 1 else 0
            y_offset = sz if i >= 2 else 0

            mosaic_image[y_offset:y_offset + sz, x_offset:x_offset + sz] = image

            for bbox, label in zip(bboxes, labels):
                xmin, ymin, xmax, ymax = bbox
                xmin = (xmin / w) * sz + x_offset
                ymin = (ymin / h) * sz + y_offset
                xmax = (xmax / w) * sz + x_offset
                ymax = (ymax / h) * sz + y_offset
                mosaic_bboxes.append([xmin, ymin, xmax, ymax])
                mosaic_labels.append(label)

        # Thu nhỏ 2x2 → 1x1 — sau đó mới normalize (transform không có Resize nữa)
        mosaic_image = cv2.resize(mosaic_image, (sz, sz))
        valid_bboxes = []
        valid_labels = []
        for bbox, label in zip(mosaic_bboxes, mosaic_labels):
            x1 = max(0.0, min(bbox[0] / 2.0, sz))
            y1 = max(0.0, min(bbox[1] / 2.0, sz))
            x2 = max(0.0, min(bbox[2] / 2.0, sz))
            y2 = max(0.0, min(bbox[3] / 2.0, sz))
            
            # Albumentations expects x1 < x2 and y1 < y2, otherwise it crashes
            if x2 > x1 and y2 > y1:
                valid_bboxes.append([x1, y1, x2, y2])
                valid_labels.append(label)
                
        mosaic_bboxes = valid_bboxes
        mosaic_labels = valid_labels

        # CHỈ apply Normalize + ToTensor, KHÔNG apply Resize/Affine
        mosaic_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                p=0.3
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_area=100,
            min_visibility=0.1
        ))

        transformed = mosaic_transform(
            image=mosaic_image, bboxes=mosaic_bboxes, class_labels=mosaic_labels
        )
        mosaic_image = transformed['image']
        mosaic_bboxes = transformed['bboxes']
        mosaic_labels = transformed['class_labels']

        return self.base.create_label_matrix(mosaic_image, mosaic_bboxes, mosaic_labels)


def get_train_transform(image_size=640):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Affine(
            translate_percent=(-0.0625, 0.0625),
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'],
        min_area=200,
        min_visibility=0.2
    ))


def get_val_transform(image_size=640):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
