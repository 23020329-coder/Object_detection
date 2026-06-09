import json
import os
import random
from collections import defaultdict

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

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
    def __init__(self, img_dir, images_info, img_to_anns, classes, transform=None, image_size=448):
        self.img_dir = img_dir
        self.images_info = images_info
        self.img_to_anns = img_to_anns
        self.classes = classes
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.transform = transform
        self.image_size = image_size
        self.S = image_size // 32
        self.C = len(classes) 

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
        label_matrix = torch.zeros((self.S, self.S, self.C + 5))

        for bbox, class_label in zip(bboxes, labels):
            xmin, ymin, xmax, ymax = bbox
            
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            width = xmax - xmin
            height = ymax - ymin
            
            # Chuẩn hóa về [0, 1]
            width /= float(self.image_size)
            height /= float(self.image_size)
            
            i = int(self.S * y_center / self.image_size)
            j = int(self.S * x_center / self.image_size)
            
            # An toàn: Tránh vượt quá kích thước lưới
            if i >= self.S: i = self.S - 1
            if j >= self.S: j = self.S - 1
            
            x_cell = (self.S * x_center / self.image_size) - j
            y_cell = (self.S * y_center / self.image_size) - i
            
            if label_matrix[i, j, self.C] == 0: 
                label_matrix[i, j, self.C] = 1.0 
                label_matrix[i, j, self.C+1 : self.C+5] = torch.tensor([x_cell, y_cell, width, height])
                label_matrix[i, j, int(class_label)] = 1.0
                
        return image, label_matrix

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
    Dataset Wrapper để tạo ảnh ghép Mosaic từ 4 ảnh (Tuyệt chiêu YOLOv4)
    """
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
            
            # Resize về kích thước gốc trước khi ghép
            image = cv2.resize(image, (sz, sz))
            
            x_offset = sz if i % 2 == 1 else 0
            y_offset = sz if i >= 2 else 0
            
            mosaic_image[y_offset:y_offset+sz, x_offset:x_offset+sz] = image
            
            for bbox, label in zip(bboxes, labels):
                xmin, ymin, xmax, ymax = bbox
                # Tỷ lệ lại tọa độ box theo kích thước ghép
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
            transformed = self.base.transform(image=mosaic_image, bboxes=mosaic_bboxes, class_labels=mosaic_labels)
            mosaic_image = transformed['image']
            mosaic_bboxes = transformed['bboxes']
            mosaic_labels = transformed['class_labels']
            
        return self.base.create_label_matrix(mosaic_image, mosaic_bboxes, mosaic_labels)

def get_train_transform(image_size=448):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5), 
        A.ToGray(p=0.2), # Thêm ảnh xám (Mô phỏng camera hồng ngoại/thiếu sáng)
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def get_val_transform(image_size=448):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
