# My Object Detection

Dự án YOLO object detection với backbone ResNet18.

## Cấu trúc thư mục

- `utils/`: Chứa các tiện ích xử lý dữ liệu, dataset, hàm loss, và đánh giá mô hình.
- `models/`: Thư mục để lưu trữ các model checkpoint (`.pth`).
- `model_arch.py`: Định nghĩa kiến trúc mô hình (YoloResNet).
- `train.py`: Vòng lặp huấn luyện mô hình.
- `predict.py`: Hàm dự đoán cho ảnh và sinh file predictions.json.

## Cách sử dụng

1. **Cài đặt thư viện**:
```bash
pip install -r requirements.txt
```

2. **Huấn luyện**:
Cập nhật các đường dẫn file JSON và thư mục ảnh trong `train.py`, sau đó chạy:
```bash
python train.py
```

3. **Dự đoán**:
Cập nhật thông tin về class và weights trong `predict.py`, sau đó sử dụng hàm `generate_predictions_json` hoặc `predict_image`.
```bash
python predict.py
```
