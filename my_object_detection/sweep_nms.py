import torch
import os
import argparse
from model_arch import YoloResNet
from utils.metrics import evaluate_model_map

def sweep(args):
    model = YoloResNet(num_classes=5)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"Bắt đầu quét siêu tham số trên tập Validation: {args.val_data}")
    print("-" * 50)

    for iou_thr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        for conf_thr in [0.01, 0.03, 0.05, 0.1]:
            print(f"Đang đánh giá với conf={conf_thr}, iou={iou_thr}...")
            result, _ = evaluate_model_map(
                model=model,
                gt_json_path=args.val_data,
                image_dir=args.val_image_dir,
                image_size=640,
                threshold=conf_thr,
                nms_iou_threshold=iou_thr,
                map_iou_threshold=0.5
            )
            map50 = result['mAP@0.5']
            chair_ap = result['per_class']['chair']['ap']
            print(f"==> Kết quả: conf={conf_thr} | iou={iou_thr} | mAP@0.5={map50:.4f} | chair_ap={chair_ap:.4f}")
            print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn đến file best.pth")
    parser.add_argument("--val_data", type=str, required=True, help="Đường dẫn đến val.json")
    parser.add_argument("--val_image_dir", type=str, required=True, help="Thư mục ảnh val")
    args = parser.parse_args()
    sweep(args)
