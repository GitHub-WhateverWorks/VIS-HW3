#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tifffile
from pycocotools import mask as mask_utils
from mmdet.apis import init_detector

DATA_DIR = Path("/content/hw3_data")
TRAIN_ROOT = DATA_DIR / "train"
TEST_ROOT = DATA_DIR / "test_release"
TEST_ID_JSON = DATA_DIR / "test_image_name_to_ids.json"

OUTPUT_DIR = Path("/content/drive/MyDrive/vis_hw3/output_exp")
SWIN_DIR = OUTPUT_DIR / "swin"
COCO_DIR = SWIN_DIR / "coco"
WORK_DIR = SWIN_DIR / "work"

CONFIG_PATH = SWIN_DIR / "_runtime_swin_config.py"
TEST_COCO_JSON = SWIN_DIR / "test_coco.json"
TRAIN_JSON = COCO_DIR / "instances_train.json"
VAL_JSON = COCO_DIR / "instances_val.json"
SUBMISSION_JSON = OUTPUT_DIR / "swin_submission.json"

CLASS_NAMES = ["class1", "class2", "class3", "class4"]
NUM_CLASSES = 4

SEED = 42
VAL_RATIO = 0.05
MIN_MASK_AREA = 3

NUM_EPOCHS = 30
BATCH_SIZE = 1
NUM_WORKERS = 0
BASE_LR = 2e-5
IMG_SCALE_LONG = 1024

SCORE_THRESH = 0.001
NMS_IOU_THRESH = 0.70
MAX_DETECTIONS_PER_IMAGE = 300
MASK_THRESH = 0.5


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SWIN_DIR.mkdir(parents=True, exist_ok=True)
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: List[str]) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    print("=" * 80, flush=True)
    print("Running command", flush=True)
    print("=" * 80, flush=True)
    print(" ".join(cmd), flush=True)
    print("=" * 80, flush=True)

    subprocess.run(cmd, check=True, env=env)


def get_mmdet_tool(name: str) -> Path:
    tool = (
        Path(sys.executable).parent.parent
        / f"lib/python3.10/site-packages/mmdet/.mim/tools/{name}.py"
    )
    if not tool.exists():
        raise FileNotFoundError(f"Cannot find MMDetection tool: {tool}")
    return tool


def read_tif_shape(path: Path) -> tuple[int, int]:
    img = tifffile.imread(str(path))
    img = np.asarray(img)
    img = np.squeeze(img)

    if img.ndim == 2:
        return int(img.shape[0]), int(img.shape[1])

    if img.ndim == 3:
        if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            return int(img.shape[1]), int(img.shape[2])
        return int(img.shape[0]), int(img.shape[1])

    raise ValueError(f"Unsupported image shape for {path}: {img.shape}")


def binary_mask_to_bbox(mask: np.ndarray) -> List[float]:
    ys, xs = np.where(mask)
    return [
        float(xs.min()),
        float(ys.min()),
        float(xs.max() - xs.min() + 1),
        float(ys.max() - ys.min() + 1),
    ]


def encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def resolve_checkpoint(pattern: str | Path) -> Path:
    pattern = str(pattern)

    if "*" not in pattern:
        p = Path(pattern)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    parent = Path(pattern).parent
    name = Path(pattern).name
    matches = sorted(parent.glob(name), key=lambda p: p.stat().st_mtime)

    if not matches:
        raise FileNotFoundError(f"No checkpoint matches: {pattern}")

    return matches[-1]


def print_model_stats(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print("=" * 80, flush=True)
    print("Model parameters", flush=True)
    print("=" * 80, flush=True)
    print(f"Trainable: {trainable / 1e6:.2f}M", flush=True)
    print(f"Total:     {total / 1e6:.2f}M", flush=True)
    print("=" * 80, flush=True)

    if trainable >= 200_000_000:
        raise RuntimeError(f"Trainable params exceed 200M: \
                           {trainable / 1e6:.2f}M")


def build_records(
    train_root: Path, min_area: int = MIN_MASK_AREA
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    sample_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()])

    image_id = 1
    ann_id = 1

    for si, sample_dir in enumerate(sample_dirs, start=1):
        if si == 1 or si % 20 == 0 or si == len(sample_dirs):
            print(
                f"building COCO {si}/{len(sample_dirs)}: \
                    {sample_dir.name}", flush=True
            )

        img_path = sample_dir / "image.tif"
        if not img_path.exists():
            continue

        h, w = read_tif_shape(img_path)

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

                rec["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": binary_mask_to_bbox(binary),
                        "area": float(area),
                        "segmentation": encode_binary_mask(binary),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        records.append(rec)
        image_id += 1

    return records


def records_to_coco(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = []
    annotations = []

    for rec in records:
        images.append(
            {
                "id": int(rec["image_id"]),
                "file_name": rec["file_name"],
                "height": int(rec["height"]),
                "width": int(rec["width"]),
            }
        )

        for ann in rec["annotations"]:
            annotations.append(dict(ann))

    categories = [{"id": i + 1, "name": name} for i,
                  name in enumerate(CLASS_NAMES)]

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def prepare_coco_dataset() -> None:
    records = build_records(TRAIN_ROOT, min_area=MIN_MASK_AREA)

    if len(records) == 0:
        raise RuntimeError("No training records found.")

    random.Random(SEED).shuffle(records)

    val_n = max(1, int(round(len(records) * VAL_RATIO)))
    val_records = records[:val_n]
    train_records = records[val_n:]

    with open(TRAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(records_to_coco(train_records), f)

    with open(VAL_JSON, "w", encoding="utf-8") as f:
        json.dump(records_to_coco(val_records), f)

    print("=" * 80, flush=True)
    print("COCO conversion complete", flush=True)
    print("=" * 80, flush=True)
    print(f"Total images:     {len(records)}", flush=True)
    print(f"Train images:     {len(train_records)}", flush=True)
    print(f"Val images:       {len(val_records)}", flush=True)
    print(
        f"Train instances:  \
            {sum(len(r['annotations']) for r in train_records)}",
        flush=True,
    )
    print(
        f"Val instances:    {sum(len(r['annotations']) for r in val_records)}",
        flush=True,
    )
    print(f"Train JSON:       {TRAIN_JSON}", flush=True)
    print(f"Val JSON:         {VAL_JSON}", flush=True)
    print("=" * 80, flush=True)


def write_swin_config() -> None:
    persistent_workers = "False" if NUM_WORKERS == 0 else "True"

    cfg = f"""
_base_ = "mmdet::swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py"

num_classes = {NUM_CLASSES}

metainfo = dict(
    classes={tuple(CLASS_NAMES)!r},
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 60, 100)]
)

data_root = ""

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_roi_extractor=dict(
            roi_layer=dict(
                output_size=14,
                sampling_ratio=0,
                type='RoIAlign',
            ),
        ),
        mask_head=dict(
            num_classes=num_classes,
        ),
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr={SCORE_THRESH},
            nms=dict(type="soft_nms", iou_threshold={NMS_IOU_THRESH}),
            max_per_img={MAX_DETECTIONS_PER_IMAGE},
            mask_thr_binary={MASK_THRESH},
        )
    )
)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="RandomChoiceResize",
        scales=[
            (512, {IMG_SCALE_LONG}),
            (576, {IMG_SCALE_LONG}),
            (640, {IMG_SCALE_LONG}),
            (704, {IMG_SCALE_LONG}),
        ],
        keep_ratio=True,
    ),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="PackDetInputs"),
]

val_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=(640, {IMG_SCALE_LONG}), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id",
          "img_path",
          "ori_shape", "img_shape", "scale_factor"),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=(640, {IMG_SCALE_LONG}), keep_ratio=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id",
          "img_path",
          "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size={BATCH_SIZE},
    num_workers={NUM_WORKERS},
    persistent_workers={persistent_workers},
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="{str(TRAIN_JSON.resolve())}",
        data_prefix=dict(img=""),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline,
        backend_args=None,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers={NUM_WORKERS},
    persistent_workers={persistent_workers},
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="{str(VAL_JSON.resolve())}",
        data_prefix=dict(img=""),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=None,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file="{str(VAL_JSON.resolve())}",
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=None,
)

test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop",
 max_epochs={NUM_EPOCHS}, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW",
      lr={BASE_LR},
      betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={{
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }}
    ),
    clip_grad=dict(max_norm=10.0, norm_type=2),
)

param_scheduler = [
    dict(type="LinearLR",
      start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type="CosineAnnealingLR",
        eta_min=1e-7,
        begin=1,
        end={NUM_EPOCHS},
        T_max={NUM_EPOCHS - 1},
        by_epoch=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_best="coco/segm_mAP_50",
        rule="greater",
        max_keep_ckpts=3,
    ),
    logger=dict(type="LoggerHook", interval=20),
)

log_processor = dict(type="LogProcessor", window_size=20, by_epoch=True)
work_dir = "{str(WORK_DIR.resolve())}"

randomness = dict(seed={SEED}, deterministic=False)
env_cfg = dict(cudnn_benchmark=True)
"""

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(cfg)

    print(f"Wrote config: {CONFIG_PATH}", flush=True)


def write_test_coco() -> None:
    with open(TEST_ID_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if isinstance(meta, dict) and "images" in meta:
        meta = meta["images"]

    images = []

    for item in meta:
        fname = item["file_name"]
        image_id = int(item["id"])
        h = int(item["height"])
        w = int(item["width"])

        img_path = TEST_ROOT / fname
        if not img_path.exists():
            raise FileNotFoundError(img_path)

        images.append(
            {
                "id": image_id,
                "file_name": str(img_path.resolve()),
                "height": h,
                "width": w,
            }
        )

    categories = [{"id": i + 1, "name": name} for i,
                  name in enumerate(CLASS_NAMES)]

    test_coco = {
        "images": images,
        "annotations": [],
        "categories": categories,
    }

    with open(TEST_COCO_JSON, "w", encoding="utf-8") as f:
        json.dump(test_coco, f)

    print("=" * 80, flush=True)
    print("Test COCO JSON generated", flush=True)
    print("=" * 80, flush=True)
    print(f"Test images: {len(images)}", flush=True)
    print(f"Test JSON:   {TEST_COCO_JSON}", flush=True)
    print("=" * 80, flush=True)


def write_infer_config(checkpoint: Path) -> Path:
    infer_config = SWIN_DIR / "swin_maskrcnn_infer_config.py"
    output_prefix = SWIN_DIR / "submission_raw"
    persistent_workers = "False" if NUM_WORKERS == 0 else "True"

    cfg = f"""
_base_ = "{str(CONFIG_PATH.resolve())}"

load_from = "{str(checkpoint.resolve())}"

model = dict(
    test_cfg=dict(
        rcnn=dict(
            score_thr={SCORE_THRESH},
            nms=dict(type="nms", iou_threshold={NMS_IOU_THRESH}),
            max_per_img={MAX_DETECTIONS_PER_IMAGE},
            mask_thr_binary={MASK_THRESH},
        )
    )
)

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=(768, 1280), keep_ratio=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id",
          "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

test_dataloader = dict(
    batch_size=1,
    num_workers={NUM_WORKERS},
    persistent_workers={persistent_workers},
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        data_root="",
        ann_file="{str(TEST_COCO_JSON.resolve())}",
        data_prefix=dict(img=""),
        metainfo=dict(classes={tuple(CLASS_NAMES)!r}),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None,
    ),
)

test_evaluator = dict(
    type="CocoMetric",
    ann_file="{str(TEST_COCO_JSON.resolve())}",
    metric=["bbox", "segm"],
    format_only=True,
    outfile_prefix="{str(output_prefix.resolve())}",
    backend_args=None,
)
"""

    with open(infer_config, "w", encoding="utf-8") as f:
        f.write(cfg)

    print(f"Wrote inference config: {infer_config}", flush=True)
    return infer_config


def train() -> None:
    prepare_coco_dataset()
    write_swin_config()

    print("=" * 80, flush=True)
    print("Building Swin Mask R-CNN for parameter check", flush=True)
    print("=" * 80, flush=True)

    model = init_detector(str(CONFIG_PATH), device="cpu")
    print_model_stats(model)

    train_tool = get_mmdet_tool("train")

    run_cmd(
        [
            sys.executable,
            str(train_tool),
            str(CONFIG_PATH),
            "--work-dir",
            str(WORK_DIR),
        ]
    )


def infer() -> None:
    write_test_coco()
    checkpoint = resolve_checkpoint(
        WORK_DIR / "best_coco_segm_mAP_50_epoch_*.pth"
    )
    infer_config = write_infer_config(checkpoint)
    test_tool = get_mmdet_tool("test")

    run_cmd(
        [
            sys.executable,
            str(test_tool),
            str(infer_config),
            str(checkpoint),
        ]
    )

    segm_json = SWIN_DIR / "submission_raw.segm.json"
    bbox_json = SWIN_DIR / "submission_raw.bbox.json"

    if not segm_json.exists():
        candidates = sorted(SWIN_DIR.glob("submission_raw*.json"))
        raise FileNotFoundError(f"Cannot find {segm_json}.\
                                 Candidates: {candidates}")

    with open(segm_json, "r", encoding="utf-8") as f:
        segm_results = json.load(f)

    bbox_results = []
    if bbox_json.exists():
        with open(bbox_json, "r", encoding="utf-8") as f:
            bbox_results = json.load(f)

    final_results = []

    if len(segm_results) > 0 and "bbox" in segm_results[0]:
        for r in segm_results:
            if "segmentation" not in r:
                continue

            seg = r["segmentation"]

            final_results.append(
                {
                    "image_id": int(r["image_id"]),
                    "bbox": [float(x) for x in r["bbox"]],
                    "score": float(r["score"]),
                    "category_id": int(r["category_id"]),
                    "segmentation": {
                        "size": [int(x) for x in seg["size"]],
                        "counts": seg["counts"],
                    },
                }
            )

    else:
        if len(bbox_results) != len(segm_results):
            raise RuntimeError(
                f"bbox/segm result length mismatch: "
                f"bbox={len(bbox_results)}, segm={len(segm_results)}"
            )

        for b, s in zip(bbox_results, segm_results):
            seg = s["segmentation"]

            final_results.append(
                {
                    "image_id": int(s["image_id"]),
                    "bbox": [float(x) for x in b["bbox"]],
                    "score": float(s["score"]),
                    "category_id": int(s["category_id"]),
                    "segmentation": {
                        "size": [int(x) for x in seg["size"]],
                        "counts": seg["counts"],
                    },
                }
            )

    with open(SUBMISSION_JSON, "w", encoding="utf-8") as f:
        json.dump(final_results, f)

    print("=" * 80, flush=True)
    print("Submission generated", flush=True)
    print("=" * 80, flush=True)
    print(f"Checkpoint:        {checkpoint}", flush=True)
    print(f"Raw segm JSON:     {segm_json}", flush=True)
    print(f"Raw bbox JSON:     {bbox_json}", flush=True)
    print(f"Submission JSON:   {SUBMISSION_JSON}", flush=True)
    print(f"Total predictions: {len(final_results)}", flush=True)
    print("=" * 80, flush=True)


def val() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Cannot find config: {CONFIG_PATH}")

    checkpoint = resolve_checkpoint(
        WORK_DIR / "best_coco_segm_mAP_50_epoch_*.pth"
    )
    test_tool = get_mmdet_tool("test")

    run_cmd(
        [
            sys.executable,
            str(test_tool),
            str(CONFIG_PATH),
            str(checkpoint),
        ]
    )


def show_best() -> None:
    print("=" * 80)
    print("Available best checkpoints")
    print("=" * 80)

    for p in sorted(WORK_DIR.glob("best_coco_segm_mAP_50_epoch_*.pth")):
        print(p)

    print("=" * 80)
    print("Recent AP50 log lines")
    print("=" * 80)

    for log_file in sorted(WORK_DIR.rglob("*.log"))[-3:]:
        print(f"\n--- {log_file} ---")
        text = log_file.read_text(errors="ignore").splitlines()
        for line in text:
            if "segm_mAP_50" in line or "coco/segm_mAP_50" in line:
                print(line)


def main() -> None:
    seed_everything(SEED)
    ensure_dirs()

    mode = "train"
    if len(sys.argv) >= 2:
        mode = sys.argv[1].lower().strip()

    print("=" * 80, flush=True)
    print("Swin Mask R-CNN training / inference", flush=True)
    print("=" * 80, flush=True)
    print(f"Mode:       {mode}", flush=True)
    print(f"Train root: {TRAIN_ROOT.resolve()}", flush=True)
    print(f"Test root:  {TEST_ROOT.resolve()}", flush=True)
    print(f"Output dir: {OUTPUT_DIR.resolve()}", flush=True)
    print(f"Swin dir:   {SWIN_DIR.resolve()}", flush=True)
    print("=" * 80, flush=True)

    if mode == "train":
        train()
    elif mode == "infer":
        infer()
    elif mode == "val":
        val()
    elif mode == "show_best":
        show_best()
    else:
        raise ValueError("Usage: python swin_colab.py\
                          [train|infer|val|show_best]")


if __name__ == "__main__":
    main()
