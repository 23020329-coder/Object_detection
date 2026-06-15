# Object Detection Submission

This folder is the main submission package for the object detection assignment.

The project implements a custom anchor-based detector in PyTorch. It does not use
complete detection frameworks such as YOLOv5/v8, Detectron2, MMDetection, or
torchvision Faster R-CNN/SSD.

## Model Summary

- Classes: `person`, `car`, `dog`, `cat`, `chair`
- Backbone: pretrained ResNet50 feature extractor
- Neck: SPPF + FPN/PAN multi-scale feature fusion
- Heads: decoupled box/objectness/classification heads
- Anchors: dataset-fitted anchors from K-means analysis
- Loss: CIoU box loss + BCE objectness + BCE classification
- Inference: confidence filtering, class-wise NMS, optional horizontal flip TTA

## Folder Structure

```text
my_object_detection/
  model_arch.py
  train.py
  predict.py
  requirements.txt
  utils/
    anchors.py
    dataset.py
    loss.py
    metrics.py
  models/
    .gitkeep
```

Large checkpoint files are intentionally not included in the zip. The prediction
script can download `best.pth` from Hugging Face when it is missing locally.

## Install

```bash
pip install -r requirements.txt
```

## Checkpoint From Hugging Face

The trained checkpoint is hosted on Hugging Face:

```text
https://huggingface.co/Quanganh6905/my-object-detection-model
```

The file used by inference is:

```text
best.pth
```

`predict.py` uses this repository by default. If `./models/best.pth` already
exists, the local checkpoint is used. If the file is missing, it is downloaded
automatically.

To override the default repository, set:

```bash
export HF_MODEL_REPO=Quanganh6905/my-object-detection-model
export HF_MODEL_FILE=best.pth
```

For a private Hugging Face repository, also set:

```bash
export HF_TOKEN=your_huggingface_token
```

On Windows PowerShell:

```powershell
$env:HF_MODEL_REPO="Quanganh6905/my-object-detection-model"
$env:HF_MODEL_FILE="best.pth"
$env:HF_TOKEN="your_huggingface_token"
```

## Train

```bash
python my_object_detection/train.py \
  --train_data public/annotations/train.json \
  --val_data public/annotations/val.json \
  --image_dir public/train/images \
  --val_image_dir public/val/images \
  --checkpoint_dir my_object_detection/models \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-4 \
  --image_size 640 \
  --eval_interval 5 \
  --mosaic_prob 0.5 \
  --close_mosaic_epochs 15 \
  --chair_oversample 1.5 \
  --conf_threshold 0.01 \
  --nms_iou_threshold 0.5
```

The best validation checkpoint is saved to:

```text
./models/best.pth
```

If mixed precision is unstable on the current GPU, add:

```bash
--no_amp
```

## Predict

Recommended validation/test inference command:

```bash
python my_object_detection/predict.py \
  --image_dir public/val/images \
  --checkpoint my_object_detection/models/best.pth \
  --output predictions.json \
  --image_size 640 \
  --conf_thresh 0.005 \
  --iou_thresh 0.45 \
  --max_candidates 300 \
  --max_detections 100 \
  --class_conf "chair:0.015" \
  --class_nms "chair:0.45" \
  --tta
```

The output file is `predictions.json`.

## Prediction Format

```json
[
  {
    "image_id": "img_example.jpg",
    "boxes": [
      {
        "class": "person",
        "confidence": 0.91,
        "bbox": [48, 72, 210, 356]
      }
    ]
  }
]
```

Images without detections are written as:

```json
{"image_id": "img_empty.jpg", "boxes": []}
```

## Local Validation

```bash
python ../public/tools/evaluate_predictions.py \
  --ground_truth ../public/annotations/val.json \
  --predictions predictions.json \
  --output val_score.json
```
