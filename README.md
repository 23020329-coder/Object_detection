# Object Detection from Scratch

Đây là dự án Phát hiện Đối tượng (Object Detection) được xây dựng từ đầu bằng PyTorch, sử dụng cấu trúc lưới dựa trên cảm hứng từ YOLOv1 nhưng được nâng cấp để cải thiện độ chính xác (mAP).

## Điểm Nổi Bật
- **Lưới dự đoán S=14**: Tăng độ phân giải đặc trưng để bắt được vật thể nhỏ.
- **GIoU Loss**: Tối ưu hóa sai số giữa các bounding box thay vì dùng MSE truyền thống, giúp bbox hội tụ rất tốt và chính xác.
- **Focal Loss**: Khắc phục sự mất cân bằng lớp (giữa các ô có đối tượng và ô không có đối tượng) ở objectness score.
- **Data Augmentation mạnh**: Sử dụng Albumentations với ColorJitter, HorizontalFlip, ShiftScaleRotate.

## 1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python. Sau đó chạy lệnh sau để cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## 2. Cách Huấn luyện (Training)
Lệnh chạy huấn luyện với tập dữ liệu:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/ \
  --epochs 30 \
  --batch_size 16
```
Mô hình tốt nhất sẽ được lưu tự động vào `./models/best.pth`.

## 3. Cách Suy luận (Inference)
Chạy suy luận trên tập dữ liệu ảnh bằng lệnh sau:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output predictions.json
```
Kết quả dự đoán sẽ được lưu dưới định dạng chuẩn trong file `predictions.json`.

## Vị trí mô hình
Mô hình sau khi huấn luyện xong sẽ lưu tại `models/best.pth`. File suy luận `predict.py` mặc định cũng đọc trọng số từ tệp này.
