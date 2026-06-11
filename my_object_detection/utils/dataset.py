"""
Dataset cho kiến trúc YOLOv3-style Multi-Scale Anchor-Based.
Sinh 3 label tensors cho 3 detection scales (P3, P4, P5).

Cải tiến:
- Multi-Anchor Assignment: Mỗi GT box được gán cho NHIỀU anchors phù hợp (IoU > 0.25),
  thay vì chỉ 1 anchor duy nhất. Giúp tăng mạnh Recall.
- Fallback Assignment: Nếu slot tốt nhất bị chiếm, tự động đẩy sang anchor kế tiếp.
  Giải quyết vấn đề vật thể chồng lấp (person + chair cùng ô lưới).
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
    find_best_anchor, find_matching_anchors, MULTI_ANCHOR_IOU_THRESH
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
        self.anchors = ANCHORS        # list of 3 lists of (w, h) tuples
        self.strides = STRIDES        # [8, 16, 32]
        self.num_anchors = NUM_ANCHORS_PER_SCALE  # 3

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
        Sinh 3 label tensors cho 3 scales với Multi-Anchor Assignment.

        Thay đổi so với bản cũ:
        - Mỗi GT box được gán cho TẤT CẢ anchors có IoU > 0.25 (thay vì chỉ 1 anchor).
        - Nếu anchor tốt nhất đã bị chiếm, GT tự động được gán cho anchor kế tiếp.
        - Giải quyết triệt để vấn đề 2 vật thể chồng lấp trong cùng 1 ô lưới.

        Returns: image, tgt_s, tgt_m, tgt_l
            Mỗi target: [S, S, A, 5+C]
            Format: [tx, ty, tw, th, objectness, class_one_hot...]
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

            # === Multi-Anchor Assignment ===
            # Tìm tất cả anchor phù hợp, sắp xếp theo IoU giảm dần
            candidates = find_matching_anchors(w, h, MULTI_ANCHOR_IOU_THRESH)

            assigned = False
            for iou_val, scale_idx, anchor_idx in candidates:
                # Dừng nếu IoU quá thấp VÀ đã gán được ít nhất 1 anchor
                if iou_val < MULTI_ANCHOR_IOU_THRESH and assigned:
                    break

                stride = self.strides[scale_idx]
                S = self.image_size // stride
                anchor_w, anchor_h = self.anchors[scale_idx][anchor_idx]

                # Xác định ô lưới
                gj = min(int(cx / stride), S - 1)
                gi = min(int(cy / stride), S - 1)

                # Chỉ assign nếu slot còn trống
                if targets[scale_idx][gi, gj, anchor_idx, 4] == 0:
                    # Encode target offsets
                    tx = cx / stride - gj        # x offset trong ô [0, 1)
                    ty = cy / stride - gi        # y offset trong ô [0, 1)
                    tw = math.log(w / anchor_w + 1e-16)   # log scale relative to anchor
                    th = math.log(h / anchor_h + 1e-16)

                    targets[scale_idx][gi, gj, anchor_idx, 0] = tx
                    targets[scale_idx][gi, gj, anchor_idx, 1] = ty
                    targets[scale_idx][gi, gj, anchor_idx, 2] = tw
                    targets[scale_idx][gi, gj, anchor_idx, 3] = th
                    targets[scale_idx][gi, gj, anchor_idx, 4] = 1.0   # objectness
                    targets[scale_idx][gi, gj, anchor_idx, 5 + int(class_label)] = 1.0  # class
                    assigned = True

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
    """Dataset Wrapper để tạo ảnh ghép Mosaic từ 4 ảnh (Tuyệt chiêu YOLOv4)."""

    def __init__(self, base_dataset, mosaic_prob=0.5):
        self.base = base_dataset
        self.mosaic_prob = mosaic_prob

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if random.random() > self.mosaic_prob:
            return self.base[idx]

        # Lấy ngẫu nhiên thêm 3 ảnh nữa
        indices = [idx] + [random.randint(0, len(self.base) - 1) for _ in range(3)]

        sz = self.base.image_size
        mosaic_image = np.zeros((sz * 2, sz * 2, 3), dtype=np.uint8)
        mosaic_bboxes = []
        mosaic_labels = []

        for i, index in enumerate(indices):
            image, bboxes, labels = self.base.load_raw(index)
            h, w = image.shape[:2]

            # Resize về kích thước chuẩn trước khi ghép
            image = cv2.resize(image, (sz, sz))

            x_offset = sz if i % 2 == 1 else 0
            y_offset = sz if i >= 2 else 0

            mosaic_image[y_offset:y_offset + sz, x_offset:x_offset + sz] = image

            for bbox, label in zip(bboxes, labels):
                xmin, ymin, xmax, ymax = bbox
                # Scale bbox theo tỷ lệ ảnh gốc → ảnh resize
                xmin = (xmin / w) * sz + x_offset
                ymin = (ymin / h) * sz + y_offset
                xmax = (xmax / w) * sz + x_offset
                ymax = (ymax / h) * sz + y_offset

                mosaic_bboxes.append([xmin, ymin, xmax, ymax])
                mosaic_labels.append(label)

        # Thu nhỏ toàn bộ ảnh ghép 2x2 về 1x1 kích thước chuẩn
        mosaic_image = cv2.resize(mosaic_image, (sz, sz))
        for bbox in mosaic_bboxes:
            bbox[0] /= 2.0
            bbox[1] /= 2.0
            bbox[2] /= 2.0
            bbox[3] /= 2.0

        if self.base.transform:
            transformed = self.base.transform(
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
