import argparse

import torch
from tqdm import tqdm

from model_arch import YoloResNet
from utils.metrics import evaluate_model_map


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def sweep(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloResNet(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    conf_values = parse_float_list(args.conf_values)
    nms_values = parse_float_list(args.nms_values)
    jobs = [(conf_thr, nms_thr) for nms_thr in nms_values for conf_thr in conf_values]

    print(f"Device: {device}")
    print(f"Validation: {args.val_data}")
    print(f"Jobs: {len(jobs)} ({len(conf_values)} conf x {len(nms_values)} NMS IoU)")
    print(f"TTA: {'on' if args.tta else 'off'}")
    print("Tip: each job runs the full validation set once.")

    best = None
    for conf_thr, nms_thr in tqdm(jobs, desc="Sweep NMS", dynamic_ncols=True):
        result, _ = evaluate_model_map(
            model=model,
            gt_json_path=args.val_data,
            image_dir=args.val_image_dir,
            image_size=args.image_size,
            threshold=conf_thr,
            nms_iou_threshold=nms_thr,
            map_iou_threshold=args.map_iou_threshold,
            use_tta=args.tta,
        )

        map50 = result["mAP@0.5"]
        chair_ap = result["per_class"]["chair"]["ap"]
        row = {
            "conf": conf_thr,
            "nms_iou": nms_thr,
            "map": map50,
            "chair_ap": chair_ap,
        }
        if best is None or row["map"] > best["map"]:
            best = row

        print(
            f"conf={conf_thr:.4f} | nms={nms_thr:.2f} | "
            f"mAP@{args.map_iou_threshold:.2f}={map50:.4f} | chair={chair_ap:.4f}"
        )

    if best is not None:
        print(
            "Best: "
            f"conf={best['conf']:.4f} | nms={best['nms_iou']:.2f} | "
            f"mAP={best['map']:.4f} | chair={best['chair_ap']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--val_data", type=str, required=True, help="Path to val.json")
    parser.add_argument("--val_image_dir", type=str, required=True, help="Validation image directory")
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--map_iou_threshold", type=float, default=0.5)
    parser.add_argument("--conf_values", type=str, default="0.01,0.03,0.05")
    parser.add_argument("--nms_values", type=str, default="0.45,0.50,0.55")
    parser.add_argument("--tta", action="store_true", help="Evaluate with horizontal flip TTA")
    args = parser.parse_args()
    sweep(args)
