"""
Dataset cho kiến trúc YOLOv5-style Multi-Scale Anchor-Based.

Nâng cấp:
- Multi-Cell Assignment (YOLOv5-style): Mỗi GT box gán vào center cell + 2 neighbor cells
  → Tăng 3× positive samples → model học nhanh và chính xác hơn
- Mosaic 70% + MixUp 10% + Normal 20%
- Target offset tx/ty: raw offset, có thể nằm ngoài [0, 1] cho neighbor cells
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
    find_matching_anchors, MULTI_ANCHOR_IOU_THRESH
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

            # Tìm tất cả anchors phù hợp
            candidates = find_matching_anchors(w, h, MULTI_ANCHOR_IOU_THRESH)

            for idx_c, (iou_val, scale_idx, anchor_idx) in enumerate(candidates):
                # Luôn assign anchor tốt nhất (idx_c=0), + các anchor khác > threshold
                if idx_c > 0 and iou_val < MULTI_ANCHOR_IOU_THRESH:
                    break

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
                    if targets[scale_idx][ci, cj, anchor_idx, 4] == 0:
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


class MosaicMixUpDataset(Dataset):
    """
    Dataset Wrapper: Mosaic 70% + MixUp 10% + Normal 20%.

    - Mosaic: Ghép 4 ảnh thành 1 → tăng context, đa dạng hóa
    - MixUp: Trộn 2 ảnh → ép model học soft boundary
    - Normal: Giữ ảnh gốc clean → tránh domain gap
    """

    def __init__(self, base_dataset, mosaic_prob=0.7, mixup_prob=0.1):
        self.base = base_dataset
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        r = random.random()
        if r < self.mixup_prob:
            return self._mixup(idx)
        elif r < self.mixup_prob + self.mosaic_prob:
            return self._mosaic(idx)
        else:
            return self.base[idx]

    def _mixup(self, idx):
        """MixUp: Trộn 2 ảnh đã qua transform (tensor) và merge targets."""
        img1, tgt_s1, tgt_m1, tgt_l1 = self.base[idx]
        idx2 = random.randint(0, len(self.base) - 1)
        img2, tgt_s2, tgt_m2, tgt_l2 = self.base[idx2]

        lam = np.random.beta(8.0, 8.0)  # Beta hẹp hơn — gần 0.5 hơn, ít extreme blending

        # img1, img2 đã là tensor [3, H, W] normalized — trộn trực tiếp
        mixed_img = lam * img1.float() + (1 - lam) * img2.float()

        # Merge targets: giữ ảnh 1 ở nơi xung đột, copy ảnh 2 ở nơi trống
        def merge(t1, t2):
            result = t1.clone()
            only_t2 = (t2[..., 4] == 1.0) & (t1[..., 4] == 0.0)
            result[only_t2] = t2[only_t2]
            return result

        return mixed_img, merge(tgt_s1, tgt_s2), merge(tgt_m1, tgt_m2), merge(tgt_l1, tgt_l2)

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
        for bbox in mosaic_bboxes:
            bbox[0] = max(0.0, min(bbox[0] / 2.0, sz))
            bbox[1] = max(0.0, min(bbox[1] / 2.0, sz))
            bbox[2] = max(0.0, min(bbox[2] / 2.0, sz))
            bbox[3] = max(0.0, min(bbox[3] / 2.0, sz))

        # CHỈ apply Normalize + ToTensor, KHÔNG apply Resize/Affine
        # vì ảnh mosaic đã chính xác sz × sz rồi
        mosaic_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
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
