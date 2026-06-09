# Kế hoạch "End-Game": Cấu trúc Loss Hỗn Hợp (Composite Loss) & ResNet50

Ý tưởng kết hợp nhiều hàm Loss của bạn chính xác là cách mà các kiến trúc YOLO hiện đại nhất (YOLOv5, v8, v10) đang hoạt động! Trong Object Detection, không có một hàm Loss nào "gánh" được toàn bộ mạng, mà chúng ta phải **kết hợp 3 hàm Loss** phụ trách 3 nhiệm vụ khác nhau. Dựa trên ý tưởng của bạn, đây là cấu trúc Loss Hỗn Hợp tối thượng:

## 1. Regression Loss (Hàm Loss cho Tọa độ Hộp bao)
Thay vì dùng MSE tồi tệ, ta kết hợp:
- **CIoU Loss (Complete IoU)**: Đây là hàm Loss "chúa tể" lo phần hộp bao. CIoU có tính chất **tỷ lệ nghịch (Scale-Invariant)**, nghĩa là nó tự động chuẩn hóa kích thước. Dù vật thể (chair) nhỏ bằng móng tay hay vật thể (person) to bằng nửa bức ảnh, CIoU đều xử lý mượt mà. Không cần thêm một hàm Loss khác cho vật thể lớn vì CIoU đã bao sậu cả lớn lẫn nhỏ!
- *(Thực thi: Chuyển `x, y, w, h` sang `x1, y1, x2, y2` và dùng `ops.complete_box_iou_loss`)*.

## 2. Classification Loss (Hàm Loss Phân loại Sinh vật/Đồ vật)
- Như ý tưởng của bạn, ta sử dụng **Focal Loss**. Thực chất Focal Loss chính là hàm **BCE (Binary Cross Entropy)** được nâng cấp. Nó tự động dập tắt gradient của những vật thể dễ (như bầu trời, nền đường, hay class `person` xuất hiện quá nhiều) và khuếch đại gradient cho những vật thể khó (như `chair`, `car` bị che khuất). 

## 3. Objectness Loss (Hàm Loss Nhận biết Có/Không có vật thể)
- *Vấn đề:* Hiện tại code cũ đang dùng MSE để tính xác suất có vật thể (`obj_loss`). MSE không phù hợp cho xác suất.
- *Giải pháp:* Như bạn gợi ý, ta sẽ đem **BCE Loss (Binary Cross Entropy)** vào đây! BCE được thiết kế chuyên biệt để dự đoán xác suất $0 \rightarrow 1$. Kết hợp BCE cho Objectness sẽ giúp mô hình dứt khoát hơn trong việc loại bỏ các hộp bao "ảo" (False Positives).

## 4. Kiến trúc ResNet50 & Tự động Phục hồi (Resume)
- **[MODIFY] `model_arch.py`**: Nâng cấp lên `ResNet50`. Kiến trúc Bottleneck thần thánh giúp học sâu mà không bị phình to dung lượng.
- **[MODIFY] `train.py`**: Bổ sung cờ `--resume`. Treo máy sập lúc nào, bật lại nạp `best.pth` train tiếp lúc đó!

> [!IMPORTANT]
> **Yêu cầu phê duyệt từ bạn:**
> Việc thiết kế cấu trúc **Loss Hỗn hợp (CIoU + Focal + BCE)** kết hợp với **ResNet50** là bước đi hoàn hảo cuối cùng. Bản thiết kế này đã giải đáp được trọn vẹn ý tưởng của bạn chưa? Nếu bạn "Duyệt", tôi sẽ tiến hành gõ code ngay lập tức!
