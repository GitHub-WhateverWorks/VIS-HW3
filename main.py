#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tifffile
import torch
import torch.nn.functional as F_nn
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.transforms import functional as F


# ============================================================
# DIRECTORIES
# ============================================================

TRAIN_ROOT = Path("./train")
OUTPUT_DIR = Path("./output")

TRAIN_DEBUG_JSON = OUTPUT_DIR / "train_split_debug.json"
VAL_DEBUG_JSON   = OUTPUT_DIR / "val_split_debug.json"

BEST_WEIGHT      = OUTPUT_DIR / "cell_maskrcnn_best.pth"
FINAL_WEIGHT     = OUTPUT_DIR / "cell_maskrcnn_final.pth"
TRAIN_LOG_JSON   = OUTPUT_DIR / "train_log.json"


# ============================================================
# DATASET / TASK CONFIG
# ============================================================

CLASS_NAMES = ["class1", "class2", "class3", "class4"]
NUM_CLASSES = 5  # background + 4 classes

SEED = 42
VAL_RATIO = 0.15
MIN_MASK_AREA = 3


# ============================================================
# MODEL CONFIG
# ============================================================

# Run these as experiments. If worse than your 0.47 baseline, set both False.
USE_DEEPER_MASK_HEAD = True
USE_STRONGER_MASK_PREDICTOR = False


# ============================================================
# TRAINING CONFIG
# ============================================================

NUM_EPOCHS = 20
BATCH_SIZE = 1
NUM_WORKERS = 0

BASE_LR = 3e-5
WEIGHT_DECAY = 1e-4

WARMUP_EPOCHS = 2

PRINT_EVERY = 20
EVAL_EVERY_EPOCH = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# INFERENCE / VALIDATION CONFIG
# ============================================================

VAL_SCORE_THRESH = 0.001
VAL_NMS_IOU_THRESH = 0.60
MAX_DETECTIONS_PER_IMAGE = 100
MASK_THRESH = 0.5


# ============================================================
# BASIC UTILITIES
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


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


def binary_mask_to_bbox(mask: np.ndarray) -> List[float]:
    ys, xs = np.where(mask)
    return [
        float(xs.min()),
        float(ys.min()),
        float(xs.max() - xs.min() + 1),
        float(ys.max() - ys.min() + 1),
    ]


def xywh_to_xyxy(box: List[float]) -> List[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def print_model_stats(model: torch.nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable: {trainable / 1e6:.2f}M  |  Total: {total / 1e6:.2f}")

    if trainable >= 200_000_000:
        raise RuntimeError(f"Trainable params exceed 200M: {trainable}")


# ============================================================
# COCO HELPERS
# ============================================================

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


def save_debug_coco(records: List[Dict[str, Any]], out_path: Path) -> None:
    gt = records_to_coco_gt(records)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gt, f)


# ============================================================
# DATASET BUILDING
# ============================================================

def build_records(train_root: Path, min_area: int = MIN_MASK_AREA) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    sample_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()])

    image_id = 1
    ann_id = 1

    for sample_dir in sample_dirs:
        img_path = sample_dir / "image.tif"
        if not img_path.exists():
            continue

        image = read_tif_as_uint8(img_path)
        h, w = image.shape[:2]

        rec: Dict[str, Any] = {
            "image_id": image_id,
            "file_name": str(img_path.resolve()),
            "height": h,
            "width": w,
            "annotations": [],
        }

        for class_id in range(1, 5):
            mask_path = sample_dir / f"class{class_id}.tif"
            if not mask_path.exists():
                continue

            mask = tifffile.imread(str(mask_path))
            mask = np.squeeze(np.asarray(mask))

            if mask.shape != (h, w):
                raise ValueError(
                    f"Shape mismatch in {sample_dir}: image={(h, w)}, "
                    f"{mask_path.name}={mask.shape}"
                )

            for inst_id in np.unique(mask):
                if float(inst_id) == 0.0:
                    continue

                binary = mask == inst_id
                area = int(binary.sum())

                if area < min_area:
                    continue

                rec["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": binary_mask_to_bbox(binary),
                    "area": float(area),
                    "segmentation": encode_binary_mask(binary),
                    "iscrowd": 0,
                })
                ann_id += 1

        records.append(rec)
        image_id += 1

    return records


# ============================================================
# AUGMENTATION HELPERS
# ============================================================

def rotate90_image_target(
    image: torch.Tensor,
    target: Dict[str, torch.Tensor],
    k: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    image = torch.rot90(image, k, dims=[1, 2])

    if target["boxes"].numel() == 0:
        return image, target

    masks = torch.rot90(target["masks"], k, dims=[1, 2])
    target["masks"] = masks

    new_boxes = []
    for m in masks:
        ys, xs = torch.where(m > 0)
        if len(xs) == 0:
            new_boxes.append(torch.tensor([0.0, 0.0, 1.0, 1.0], device=image.device))
        else:
            new_boxes.append(torch.stack([
                xs.float().min(),
                ys.float().min(),
                xs.float().max(),
                ys.float().max(),
            ]))

    target["boxes"] = torch.stack(new_boxes, dim=0)
    return image, target


def random_scale_jitter(
    image: torch.Tensor,
    target: Dict[str, torch.Tensor],
    min_scale: float = 0.9,
    max_scale: float = 1.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    scale = random.uniform(min_scale, max_scale)
    _, h, w = image.shape

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    image = F_nn.interpolate(
        image.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    if target["masks"].numel() > 0:
        masks = target["masks"].float().unsqueeze(0)
        masks = F_nn.interpolate(
            masks,
            size=(new_h, new_w),
            mode="nearest",
        ).squeeze(0)

        target["masks"] = (masks > 0.5).to(torch.uint8)

        boxes = target["boxes"] * scale
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, new_w - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, new_h - 1)
        target["boxes"] = boxes

    return image, target


# ============================================================
# DATASET
# ============================================================

class CellInstanceDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], is_train: bool):
        self.records = records
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rec = self.records[idx]
        image_np = read_tif_as_uint8(rec["file_name"])
        image = F.to_tensor(image_np)

        boxes: List[List[float]] = []
        labels: List[int] = []
        masks: List[np.ndarray] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in rec["annotations"]:
            mask = mask_utils.decode(ann["segmentation"]).astype(bool)
            boxes.append(xywh_to_xyxy(ann["bbox"]))
            labels.append(int(ann["category_id"]))
            masks.append(mask.astype(np.uint8))
            areas.append(float(ann["area"]))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, rec["height"], rec["width"]), dtype=torch.uint8)
            areas_t = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            masks_t = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            areas_t = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd_t = torch.as_tensor(iscrowd, dtype=torch.int64)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([int(rec["image_id"])], dtype=torch.int64),
            "area": areas_t,
            "iscrowd": iscrowd_t,
        }

        if self.is_train:
            image, target = self._augment(image, target)

        return image, target

    def _augment(
        self,
        image: torch.Tensor,
        target: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _, h, w = image.shape

        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            if target["boxes"].numel() > 0:
                boxes = target["boxes"]
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = w - x2
                boxes[:, 2] = w - x1
                target["boxes"] = boxes
                target["masks"] = torch.flip(target["masks"], dims=[2])

        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            if target["boxes"].numel() > 0:
                boxes = target["boxes"]
                y1 = boxes[:, 1].clone()
                y2 = boxes[:, 3].clone()
                boxes[:, 1] = h - y2
                boxes[:, 3] = h - y1
                target["boxes"] = boxes
                target["masks"] = torch.flip(target["masks"], dims=[1])

        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            image, target = rotate90_image_target(image, target, k)

        if random.random() < 0.5:
            image, target = random_scale_jitter(image, target, 0.9, 1.1)

        if random.random() < 0.2:
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            image = F.adjust_brightness(image, brightness)
            image = F.adjust_contrast(image, contrast)

        if random.random() < 0.05:
            sigma = random.uniform(0.1, 1.0)
            image = F.gaussian_blur(
                image,
                kernel_size=[5, 5],
                sigma=[sigma, sigma],
            )

        image = image.clamp(0.0, 1.0)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model() -> torch.nn.Module:
    base = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,

        box_score_thresh=VAL_SCORE_THRESH,
        box_nms_thresh=VAL_NMS_IOU_THRESH,
        box_detections_per_img=MAX_DETECTIONS_PER_IMAGE,

        rpn_pre_nms_top_n_train=1000,
        rpn_post_nms_top_n_train=500,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=500,
    )

    in_features = base.roi_heads.box_predictor.cls_score.in_features
    base.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES,
    )

    if USE_DEEPER_MASK_HEAD:
        base.roi_heads.mask_head = MaskRCNNHeads(
            256,
            layers=(256, 256, 256, 256, 256, 256),
            dilation=1,
        )

    in_channels = base.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 512 if USE_STRONGER_MASK_PREDICTOR else 256

    base.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels,
        hidden_layer,
        NUM_CLASSES,
    )

    for name, param in base.backbone.body.named_parameters():
        if any(name.startswith(lyr) for lyr in ("conv1", "bn1", "layer1")):
            param.requires_grad = False

    return base


# ============================================================
# COCO EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_coco(
    model: torch.nn.Module,
    loader: DataLoader,
    val_records: List[Dict[str, Any]],
    device: str,
) -> Dict[str, float]:
    model.eval()

    gt_dict = records_to_coco_gt(val_records)
    pred_results = []

    for bi, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            image_id = int(tgt["image_id"].item())

            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            masks = (out["masks"].detach().cpu().numpy()[:, 0] >= MASK_THRESH)

            keep = scores >= VAL_SCORE_THRESH
            scores = scores[keep]
            labels = labels[keep]
            masks = masks[keep]

            order = np.argsort(-scores)[:MAX_DETECTIONS_PER_IMAGE]

            for idx in order:
                category_id = int(labels[idx])
                if not (1 <= category_id <= 4):
                    continue

                rle = mask_utils.encode(np.asfortranarray(masks[idx].astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")

                pred_results.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "score": float(scores[idx]),
                })

        print(f"evaluated image {bi:03d}/{len(loader):03d}", flush=True)

    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    if len(pred_results) == 0:
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


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: str,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.train()

    epoch_start = time.time()
    running_loss = 0.0
    num_batches = len(loader)
    loss_sums: Dict[str, float] = {}

    for step, (images, targets) in enumerate(loader, start=1):

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        loss_value = float(losses.detach().cpu().item())
        running_loss += loss_value

        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach().cpu().item())

        if step % PRINT_EVERY == 0 or step == 1 or step == num_batches:
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"CUDA mem allocated={alloc:.2f}GB reserved={reserved:.2f}GB", flush=True)
                print(
                    f"image shape={tuple(images[0].shape)} "
                    f"instances={len(targets[0]['labels'])}",
                    flush=True,
                )
            avg_loss = running_loss / step
            print(
                f"[Epoch {epoch:03d}/{total_epochs:03d}] "
                f"step {step:04d}/{num_batches:04d} | "
                f"loss={loss_value:.4f} | avg={avg_loss:.4f}",
                flush=True,
            )

    epoch_time = time.time() - epoch_start
    avg_loss = running_loss / max(num_batches, 1)

    out = {
        "loss_total": avg_loss,
        "epoch_time_sec": epoch_time,
    }

    for k, v in loss_sums.items():
        out[k] = v / max(num_batches, 1)

    return out


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.05,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-7,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )

    return scheduler


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    seed_everything(SEED)

    if not TRAIN_ROOT.exists():
        raise FileNotFoundError(f"Cannot find train folder: {TRAIN_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Mask R-CNN ResNet50-FPN training")
    print("=" * 80)
    print(f"Device:        {DEVICE}")
    print(f"Train root:    {TRAIN_ROOT.resolve()}")
    print(f"Output dir:    {OUTPUT_DIR.resolve()}")
    print(f"Epochs:        {NUM_EPOCHS}  (warmup={WARMUP_EPOCHS})")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"AdamW LR:      {BASE_LR}")
    print(f"Deeper mask:   {USE_DEEPER_MASK_HEAD}")
    print(f"Strong mask:   {USE_STRONGER_MASK_PREDICTOR}")
    print("=" * 80)

    all_records = build_records(TRAIN_ROOT, min_area=MIN_MASK_AREA)

    if len(all_records) == 0:
        raise RuntimeError("No training images found.")

    random.Random(SEED).shuffle(all_records)

    val_n = max(1, int(round(len(all_records) * VAL_RATIO)))
    val_records = all_records[:val_n]
    train_records = all_records[val_n:]

    save_debug_coco(train_records, TRAIN_DEBUG_JSON)
    save_debug_coco(val_records, VAL_DEBUG_JSON)

    print(f"Total: {len(all_records)} | Train: {len(train_records)} | Val: {len(val_records)}")
    print(f"Train instances: {sum(len(r['annotations']) for r in train_records)}")
    print(f"Val instances:   {sum(len(r['annotations']) for r in val_records)}")

    train_loader = DataLoader(
        CellInstanceDataset(train_records, is_train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    val_loader = DataLoader(
        CellInstanceDataset(val_records, is_train=False),
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    model = build_model()
    model.to(DEVICE)
    print_model_stats(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = build_scheduler(
        optimizer,
        WARMUP_EPOCHS,
        NUM_EPOCHS,
    )

    history: List[Dict[str, Any]] = []
    best_ap50 = -1.0
    total_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        epoch_info = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=DEVICE,
            epoch=epoch,
            total_epochs=NUM_EPOCHS,
        )

        scheduler.step()

        do_eval = (epoch % EVAL_EVERY_EPOCH == 0) or (epoch == NUM_EPOCHS)

        ap = None
        ap50 = None
        ap75 = None

        if do_eval:
            eval_start = time.time()

            coco_metrics = evaluate_coco(
                model=model,
                loader=val_loader,
                val_records=val_records,
                device=DEVICE,
            )

            ap = coco_metrics["AP"]
            ap50 = coco_metrics["AP50"]
            ap75 = coco_metrics["AP75"]

            eval_time = time.time() - eval_start

            print(
                f"[Val] epoch={epoch:03d}  "
                f"mask_AP={ap:.4f}  "
                f"mask_AP50={ap50:.4f}  "
                f"mask_AP75={ap75:.4f}  "
                f"eval_time={format_seconds(eval_time)}"
            )

            if ap50 > best_ap50:
                best_ap50 = ap50

                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_ap50": best_ap50,
                    "config": {
                        "num_classes": NUM_CLASSES,
                        "class_names": CLASS_NAMES,
                        "backbone": "maskrcnn_resnet50_fpn",
                        "cascade": False,
                        "deeper_mask_head": USE_DEEPER_MASK_HEAD,
                        "stronger_mask_predictor": USE_STRONGER_MASK_PREDICTOR,
                        "score_thresh": VAL_SCORE_THRESH,
                        "nms_iou_thresh": VAL_NMS_IOU_THRESH,
                        "max_detections": MAX_DETECTIONS_PER_IMAGE,
                        "mask_thresh": MASK_THRESH,
                    },
                }, BEST_WEIGHT)

                print(f"Saved best: {BEST_WEIGHT}  AP50={best_ap50:.4f}")

        elapsed = time.time() - total_start
        avg_ep_time = elapsed / epoch
        eta = avg_ep_time * (NUM_EPOCHS - epoch)
        lr_now = optimizer.param_groups[0]["lr"]

        log_item: Dict[str, Any] = {
            "epoch": epoch,
            "lr": lr_now,
            "best_ap50": best_ap50,
            **epoch_info,
        }

        if ap50 is not None:
            log_item["val_ap"] = ap
            log_item["val_ap50"] = ap50
            log_item["val_ap75"] = ap75

        history.append(log_item)

        with open(TRAIN_LOG_JSON, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(
            "\n" + "=" * 80 + "\n"
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"loss={epoch_info['loss_total']:.4f} | "
            f"lr={lr_now:.2e} | "
            f"best_AP50={best_ap50:.4f} | "
            f"ETA={format_seconds(eta)}\n"
            + "=" * 80 + "\n"
        )

    torch.save({
        "model_state": model.state_dict(),
        "epoch": NUM_EPOCHS,
        "best_ap50": best_ap50,
        "config": {
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "backbone": "maskrcnn_resnet50_fpn",
            "cascade": False,
            "deeper_mask_head": USE_DEEPER_MASK_HEAD,
            "stronger_mask_predictor": USE_STRONGER_MASK_PREDICTOR,
            "score_thresh": VAL_SCORE_THRESH,
            "nms_iou_thresh": VAL_NMS_IOU_THRESH,
            "max_detections": MAX_DETECTIONS_PER_IMAGE,
            "mask_thresh": MASK_THRESH,
        },
    }, FINAL_WEIGHT)

    print(f"Final checkpoint: {FINAL_WEIGHT}")
    print(f"Best checkpoint:  {BEST_WEIGHT}")
    print(f"Training log:     {TRAIN_LOG_JSON}")


if __name__ == "__main__":
    main()