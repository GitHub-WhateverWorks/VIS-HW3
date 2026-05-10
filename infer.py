#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import main as train_mod


CLASS_NAMES = ["class1", "class2", "class3", "class4"]
NUM_CLASSES = 5
SEED = 42
VAL_RATIO = 0.15
MIN_MASK_AREA = 3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_test_fn(batch):
    images, image_ids = zip(*batch)
    return list(images), list(image_ids)


def collate_val_fn(batch):
    return tuple(zip(*batch))


def read_tif_as_uint8(path: str | Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    img = np.asarray(img)
    img = np.squeeze(img)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] >= 3:
            img = img[..., :3]
        else:
            raise ValueError(f"Unsupported image shape after loading {path}: {img.shape}")
    else:
        raise ValueError(f"Unsupported image shape for {path}: {img.shape}")

    if img.dtype == np.uint8:
        return np.ascontiguousarray(img)

    img = img.astype(np.float32)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros((*img.shape[:2], 3), dtype=np.uint8)

    lo, hi = np.percentile(img[finite], [1, 99])
    if hi <= lo:
        lo = float(np.min(img[finite]))
        hi = float(np.max(img[finite]))

    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)

    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    img = (img * 255.0 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(img)


class CellTestDataset(Dataset):
    def __init__(self, test_root: Path, mapping_json: Path):
        self.test_root = test_root

        with open(mapping_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # supports either list[...] or {"images":[...]}
        if isinstance(meta, dict) and "images" in meta:
            meta = meta["images"]

        self.samples = []

        for item in meta:
            fname = item["file_name"]
            image_id = int(item["id"])

            path = test_root / fname
            if not path.exists():
                raise FileNotFoundError(path)

            self.samples.append({
                "path": path,
                "image_id": image_id,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_np = read_tif_as_uint8(sample["path"])
        image = F.to_tensor(image_np)

        return image, sample["image_id"]


def load_model(weight_path: Path, device: str):
    model = train_mod.build_model()

    ckpt = torch.load(weight_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()

    print(f"Loaded weight: {weight_path}")
    if isinstance(ckpt, dict):
        print(f"Checkpoint epoch: {ckpt.get('epoch')}")
        print(f"Checkpoint best_ap50: {ckpt.get('best_ap50')}")

    return model


@torch.no_grad()
def generate_submission(
    model,
    loader,
    device: str,
    score_thresh: float,
    mask_thresh: float,
    max_dets: int,
    out_json: Path,
):
    model.eval()
    results: List[Dict[str, Any]] = []

    start = time.time()

    for bi, (images, image_ids) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, image_id in zip(outputs, image_ids):
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            boxes = out["boxes"].detach().cpu().numpy()
            masks = (out["masks"].detach().cpu().numpy()[:, 0] >= mask_thresh)

            keep = scores >= score_thresh
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            masks = masks[keep]

            order = np.argsort(-scores)[:max_dets]

            for idx in order:
                c = int(labels[idx])
                if not (1 <= c <= 4):
                    continue

                x1, y1, x2, y2 = boxes[idx]
                bbox = [
                    float(x1),
                    float(y1),
                    float(max(0.0, x2 - x1)),
                    float(max(0.0, y2 - y1)),
                ]

                rle = mask_utils.encode(np.asfortranarray(masks[idx].astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")

                results.append({
                    "image_id": int(image_id),
                    "category_id": c,
                    "bbox": bbox,
                    "score": float(scores[idx]),
                    "segmentation": rle,
                })

        print(f"inferred image {bi:04d}/{len(loader):04d}", flush=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)

    elapsed = time.time() - start
    print("=" * 80)
    print(f"Saved submission: {out_json}")
    print(f"Total predictions: {len(results)}")
    print(f"Inference time: {elapsed:.1f}s")
    print("=" * 80)


def records_to_coco_gt(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = []
    annotations = []
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)]

    ann_id = 1
    for rec in records:
        images.append({
            "id": int(rec["image_id"]),
            "file_name": rec["file_name"],
            "height": int(rec["height"]),
            "width": int(rec["width"]),
        })

        for ann in rec["annotations"]:
            a = dict(ann)
            a["id"] = ann_id
            a["image_id"] = int(rec["image_id"])
            a["category_id"] = int(a["category_id"])
            a["area"] = float(a["area"])
            a["iscrowd"] = int(a.get("iscrowd", 0))
            annotations.append(a)
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def build_val_records(train_root: Path, val_ratio: float, seed: int):
    records = train_mod.build_records(train_root, min_area=MIN_MASK_AREA)
    random.Random(seed).shuffle(records)
    val_n = max(1, int(round(len(records) * val_ratio)))
    return records[:val_n]


@torch.no_grad()
def predict_val_coco_results(
    model,
    loader,
    device: str,
    score_thresh: float,
    mask_thresh: float,
    max_dets: int,
):
    model.eval()
    results = []

    for bi, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            image_id = int(tgt["image_id"].item())

            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            masks = (out["masks"].detach().cpu().numpy()[:, 0] >= mask_thresh)

            keep = scores >= score_thresh
            scores = scores[keep]
            labels = labels[keep]
            masks = masks[keep]

            order = np.argsort(-scores)[:max_dets]

            for idx in order:
                c = int(labels[idx])
                if not (1 <= c <= 4):
                    continue

                rle = mask_utils.encode(np.asfortranarray(masks[idx].astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")

                results.append({
                    "image_id": image_id,
                    "category_id": c,
                    "segmentation": rle,
                    "score": float(scores[idx]),
                })

        print(f"evaluated image {bi:03d}/{len(loader):03d}", flush=True)

    return results


def evaluate_coco_segm(gt_dict: Dict[str, Any], pred_results: List[Dict[str, Any]]):
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    if len(pred_results) == 0:
        print("No predictions.")
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
        }

    coco_dt = coco_gt.loadRes(pred_results)

    evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
    evaluator.params.imgIds = sorted([img["id"] for img in gt_dict["images"]])
    evaluator.params.catIds = [1, 2, 3, 4]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return {
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AP_small": float(evaluator.stats[3]),
        "AP_medium": float(evaluator.stats[4]),
        "AP_large": float(evaluator.stats[5]),
    }


def run_val_eval(args, model):
    val_records = build_val_records(Path(args.train_root), args.val_ratio, args.seed)
    dataset = train_mod.CellInstanceDataset(val_records, is_train=False)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_val_fn,
        pin_memory=False,
    )

    gt_dict = records_to_coco_gt(val_records)
    pred_results = predict_val_coco_results(
        model=model,
        loader=loader,
        device=args.device,
        score_thresh=args.score_thresh,
        mask_thresh=args.mask_thresh,
        max_dets=args.max_dets,
    )

    metrics = evaluate_coco_segm(gt_dict, pred_results)

    out_metrics = Path(args.output_json).with_name("val_coco_metrics.json")
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 80)
    print(f"COCO mask AP:   {metrics['AP']:.4f}")
    print(f"COCO mask AP50: {metrics['AP50']:.4f}")
    print(f"COCO mask AP75: {metrics['AP75']:.4f}")
    print(f"Saved metrics: {out_metrics}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_root", type=str, default="./test_release")
    parser.add_argument("--train_root", type=str, default="./train")
    parser.add_argument("--weight", type=str, default="./output/cell_maskrcnn_best.pth")
    parser.add_argument("--output_json", type=str, default="./output/submission.json")

    parser.add_argument("--score_thresh", type=float, default=0.001)
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    parser.add_argument("--max_dets", type=int, default=300)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval_val", action="store_true")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--test_id_json",
        type=str,
        default="./test_image_name_to_ids.json"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    weight = Path(args.weight)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Mask R-CNN v2 inference / COCO evaluation")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Weight: {weight.resolve()}")
    print(f"Score threshold: {args.score_thresh}")
    print(f"Mask threshold: {args.mask_thresh}")
    print(f"Max detections/image: {args.max_dets}")
    print("=" * 80)

    model = load_model(weight, args.device)
    '''
    if args.eval_val:
        run_val_eval(args, model)
        return
    '''
    test_root = Path(args.test_root)
    dataset = CellTestDataset(
        test_root=test_root,
        mapping_json=Path(args.test_id_json),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_test_fn,
        pin_memory=False,
    )

    print(f"Test root: {test_root.resolve()}")
    print(f"Test images: {len(dataset)}")
    print(f"Output JSON: {out_json.resolve()}")
    print("=" * 80)

    generate_submission(
        model=model,
        loader=loader,
        device=args.device,
        score_thresh=args.score_thresh,
        mask_thresh=args.mask_thresh,
        max_dets=args.max_dets,
        out_json=out_json,
    )


if __name__ == "__main__":
    main()