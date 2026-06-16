#!/usr/bin/env python
"""Benchmark real video -> in-memory official LipFD input -> model inference.

This script keeps the official preprocessing semantics but avoids writing the
intermediate 1000x2500 PNG samples to disk. It is intended for deployment-style
speed diagnostics, not for replacing data/datasets.py.
"""

import argparse
import csv
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from librosa import feature as audio
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

N_EXTRACT = 10
WINDOW_LEN = 5
BIG_IMAGE_SIZE = 500
DEFAULT_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class VideoTask:
    video_path: Path
    audio_path: Path
    label: int | None
    frame_count: int | None = None


@dataclass
class WindowTensor:
    video_path: str
    window_idx: int
    label: int | None
    img: torch.Tensor
    crops: list[list[torch.Tensor]]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def add_elapsed(detail, key, start):
    if detail is not None:
        detail[key] = detail.get(key, 0.0) + time.perf_counter() - start


def load_model(args, device):
    from lightweight.models import LipFDRegionLight

    state = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = state.get("args", {}) if isinstance(state, dict) else {}
    clip_name = args.clip_name or ckpt_args.get("clip_name", "ViT-L/14")
    backbone = args.backbone or ckpt_args.get("backbone", "resnet18")

    model = LipFDRegionLight(clip_name=clip_name, backbone=backbone)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()
    return model, clip_name, backbone


def discover_tasks(args):
    if args.video_path:
        if not args.audio_path:
            raise ValueError("--audio_path is required when --video_path is used")
        return [VideoTask(Path(args.video_path), Path(args.audio_path), args.label)]

    video_root = Path(args.video_root)
    audio_root = Path(args.audio_root)
    tasks = []
    for label, dirname in [(0, "0_real"), (1, "1_fake")]:
        video_dir = video_root / dirname
        audio_dir = audio_root / dirname
        if not video_dir.exists():
            continue
        videos = sorted(
            p for p in video_dir.iterdir()
            if p.is_file() and p.suffix.lower() in DEFAULT_VIDEO_EXTS
        )
        if args.max_videos_per_class is not None:
            videos = videos[: args.max_videos_per_class]
        for video_path in videos:
            tasks.append(VideoTask(video_path, audio_dir / f"{video_path.stem}.wav", label))
    return tasks


def official_mel_rgba_in_memory(audio_file, detail=None):
    """Match plt.imsave(single_channel_mel) -> plt.imread(...) * 255 in memory."""
    start = time.perf_counter()
    data, sr = librosa.load(audio_file)
    add_elapsed(detail, "audio_load_seconds", start)

    start = time.perf_counter()
    mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)
    add_elapsed(detail, "mel_compute_seconds", start)

    start = time.perf_counter()
    buffer = io.BytesIO()
    plt.imsave(buffer, mel, format="png")
    buffer.seek(0)
    mel_img = plt.imread(buffer) * 255
    add_elapsed(detail, "mel_png_roundtrip_seconds", start)
    return mel_img.astype(np.uint8)


def select_video_frames(video_path, detail=None, num_windows=N_EXTRACT, frame_count_override=None):
    start = time.perf_counter()
    capture = cv2.VideoCapture(str(video_path))
    frame_count = (
        int(frame_count_override)
        if frame_count_override is not None and int(frame_count_override) > 0
        else int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    )
    add_elapsed(detail, "video_open_meta_seconds", start)
    if frame_count <= WINDOW_LEN:
        capture.release()
        raise ValueError(f"too few frames: {frame_count}")

    frame_idx = np.linspace(
        0,
        frame_count - WINDOW_LEN - 1,
        num_windows,
        endpoint=True,
        dtype=np.int32,
    ).tolist()
    frame_idx.sort()
    frame_sequence = [i for num in frame_idx for i in range(num, num + WINDOW_LEN)]
    wanted = set(frame_sequence)

    frame_list = []
    current_frame = 0
    while current_frame <= frame_sequence[-1]:
        start = time.perf_counter()
        ret, frame = capture.read()
        add_elapsed(detail, "frame_read_seconds", start)
        if not ret:
            capture.release()
            raise RuntimeError(f"failed reading frame {current_frame}: {video_path}")
        if current_frame in wanted:
            start = time.perf_counter()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame_list.append(cv2.resize(frame, (BIG_IMAGE_SIZE, BIG_IMAGE_SIZE)))
            add_elapsed(detail, "frame_convert_resize_seconds", start)
        current_frame += 1
    capture.release()

    if len(frame_list) != len(frame_sequence):
        raise RuntimeError(
            f"selected {len(frame_list)} frames, expected {len(frame_sequence)}: {video_path}"
        )
    return frame_count, frame_sequence, frame_list


def make_big_images(frame_count, frame_sequence, frame_list, mel_rgba, detail=None):
    mapping = mel_rgba.shape[1] / frame_count
    big_images = []
    group = 0

    for i in range(0, len(frame_list), WINDOW_LEN):
        begin = int(np.round(frame_sequence[i] * mapping))
        end = int(np.round((frame_sequence[i] + WINDOW_LEN) * mapping))
        if end <= begin:
            continue
        start = time.perf_counter()
        sub_mel = cv2.resize(
            mel_rgba[:, begin:end],
            (BIG_IMAGE_SIZE * WINDOW_LEN, BIG_IMAGE_SIZE),
        )
        add_elapsed(detail, "mel_sub_resize_seconds", start)

        start = time.perf_counter()
        frames = np.concatenate(frame_list[i : i + WINDOW_LEN], axis=1)
        add_elapsed(detail, "frame_concat_seconds", start)

        start = time.perf_counter()
        big = np.concatenate((sub_mel[:, :, :3], frames[:, :, :3]), axis=0)
        add_elapsed(detail, "big_image_concat_seconds", start)

        start = time.perf_counter()
        big_images.append((group, big.astype(np.uint8, copy=False)))
        add_elapsed(detail, "big_image_cast_seconds", start)
        group += 1
    return big_images


def dataset_like_tensor_to_window(img, label, video_path, window_idx, detail=None):
    resize_224 = transforms.Resize((224, 224))
    start = time.perf_counter()
    # Match current server data/datasets.py exactly:
    # [img[:, 500:, i:i + 500] for i in range(5)]
    # This is intentionally not i * 500.
    crops0 = [
        resize_224(img[:, BIG_IMAGE_SIZE:, i : i + BIG_IMAGE_SIZE])
        for i in range(WINDOW_LEN)
    ]
    add_elapsed(detail, "crop0_resize_seconds", start)

    crop_idx = [(28, 196), (61, 163)]
    start = time.perf_counter()
    crops1, crops2 = [], []
    for crop in crops0:
        crops1.append(
            resize_224(
                crop[
                    :,
                    crop_idx[0][0] : crop_idx[0][1],
                    crop_idx[0][0] : crop_idx[0][1],
                ]
            )
        )
        crops2.append(
            resize_224(
                crop[
                    :,
                    crop_idx[1][0] : crop_idx[1][1],
                    crop_idx[1][0] : crop_idx[1][1],
                ]
            )
        )
    add_elapsed(detail, "crop_inner_resize_seconds", start)

    start = time.perf_counter()
    crops = [crops0, crops1, crops2]
    add_elapsed(detail, "crop_pack_seconds", start)

    start = time.perf_counter()
    img = transforms.Resize((1120, 1120))(img)
    add_elapsed(detail, "full_image_resize_seconds", start)
    return WindowTensor(str(video_path), window_idx, label, img, crops)


def big_image_to_tensors(big_rgb, label, video_path, window_idx, detail=None):
    # Keep this path equivalent to the current server data/datasets.py:
    # cv2.imread-style BGR, 0-255 float range, torchvision Resize, no Normalize.
    start = time.perf_counter()
    bgr = big_rgb[:, :, ::-1].copy()
    add_elapsed(detail, "rgb_to_bgr_seconds", start)

    start = time.perf_counter()
    img = torch.tensor(bgr, dtype=torch.float32)
    img = img.permute(2, 0, 1)
    add_elapsed(detail, "tensor_from_numpy_seconds", start)
    return dataset_like_tensor_to_window(img, label, video_path, window_idx, detail=detail)


def cache_frame_tensors(frame_list, device=None, detail=None):
    start = time.perf_counter()
    if not frame_list:
        return []
    frame_bgr = np.stack([frame[:, :, 2::-1] for frame in frame_list], axis=0)
    tensor_batch = torch.from_numpy(frame_bgr).permute(0, 3, 1, 2).float()
    if device is not None and device.type != "cpu":
        tensor_batch = tensor_batch.to(device, non_blocking=True)
    tensors = [tensor_batch[i] for i in range(tensor_batch.shape[0])]
    add_elapsed(detail, "frame_tensor_cache_seconds", start)
    return tensors


def build_top_mel_tensor_batch(sub_mels, device=None, detail=None):
    """Batch build BGR 0-255 top mel tensors for all windows in one video."""
    start = time.perf_counter()
    if not sub_mels:
        return None
    top_bgr = np.stack([sub_mel[:, :, 2::-1] for sub_mel in sub_mels], axis=0)
    top_batch = torch.from_numpy(top_bgr).permute(0, 3, 1, 2).float()
    if device is not None and device.type != "cpu":
        top_batch = top_batch.to(device, non_blocking=True)
    add_elapsed(detail, "top_mel_tensor_batch_seconds", start)
    return top_batch


def build_window_tensors_batched(
    window_specs,
    top_batch,
    frame_tensors,
    label,
    video_path,
    detail=None,
):
    """Build official-equivalent tensors for many windows together.

    Frame tensors are cached per video. Crops are generated from a narrow
    frame source instead of the full 2500-pixel-wide bottom strip. Resize runs
    on the tensor device, so passing CUDA tensors enables experimental GPU
    batch resize/crop while keeping the same slicing semantics.
    """
    start = time.perf_counter()
    bottom_batch = torch.stack(
        [
            torch.cat(frame_tensors[spec["frame_offset"] : spec["frame_offset"] + WINDOW_LEN], dim=2)
            for spec in window_specs
        ]
    )
    img_batch = torch.cat((top_batch, bottom_batch), dim=2)
    add_elapsed(detail, "global_tensor_build_seconds", start)

    resize_224 = transforms.Resize((224, 224))
    resize_full = transforms.Resize((1120, 1120))

    start = time.perf_counter()
    img_batch = resize_full(img_batch)
    add_elapsed(detail, "full_image_resize_seconds", start)

    # Match current data/datasets.py exactly:
    # img[:, 500:, i:i+500] for i in range(5). For i > 0 this includes
    # the first frame plus a few columns from the second frame.
    start = time.perf_counter()
    crop_source_batch = torch.stack(
        [
            torch.cat(
                [
                    frame_tensors[spec["frame_offset"]],
                    frame_tensors[spec["frame_offset"] + 1][:, :, : WINDOW_LEN - 1],
                ],
                dim=2,
            )
            for spec in window_specs
        ]
    )
    add_elapsed(detail, "crop_source_build_seconds", start)

    start = time.perf_counter()
    crops0_batch = [
        resize_224(crop_source_batch[:, :, :, i : i + BIG_IMAGE_SIZE])
        for i in range(WINDOW_LEN)
    ]
    add_elapsed(detail, "crop0_resize_seconds", start)

    crop_idx = [(28, 196), (61, 163)]
    start = time.perf_counter()
    crops1_batch, crops2_batch = [], []
    for crop in crops0_batch:
        crops1_batch.append(
            resize_224(
                crop[
                    :,
                    :,
                    crop_idx[0][0] : crop_idx[0][1],
                    crop_idx[0][0] : crop_idx[0][1],
                ]
            )
        )
        crops2_batch.append(
            resize_224(
                crop[
                    :,
                    :,
                    crop_idx[1][0] : crop_idx[1][1],
                    crop_idx[1][0] : crop_idx[1][1],
                ]
            )
    )
    add_elapsed(detail, "crop_inner_resize_seconds", start)

    start = time.perf_counter()
    windows = []
    for batch_idx, spec in enumerate(window_specs):
        crops = [
            [crops0_batch[i][batch_idx] for i in range(WINDOW_LEN)],
            [crops1_batch[i][batch_idx] for i in range(WINDOW_LEN)],
            [crops2_batch[i][batch_idx] for i in range(WINDOW_LEN)],
        ]
        windows.append(
            WindowTensor(
                str(video_path),
                spec["window_idx"],
                label,
                img_batch[batch_idx],
                crops,
            )
        )
    add_elapsed(detail, "crop_pack_seconds", start)
    return windows


def make_window_tensors(
    frame_count,
    frame_sequence,
    frame_list,
    mel_rgba,
    label,
    video_path,
    detail=None,
    preprocess_device=None,
):
    mapping = mel_rgba.shape[1] / frame_count
    window_specs = []
    sub_mels = []
    frame_tensors = cache_frame_tensors(frame_list, device=preprocess_device, detail=detail)

    for i in range(0, len(frame_list), WINDOW_LEN):
        begin = int(np.round(frame_sequence[i] * mapping))
        end = int(np.round((frame_sequence[i] + WINDOW_LEN) * mapping))
        if end <= begin:
            continue

        start = time.perf_counter()
        sub_mels.append(cv2.resize(mel_rgba[:, begin:end], (BIG_IMAGE_SIZE * WINDOW_LEN, BIG_IMAGE_SIZE)))
        add_elapsed(detail, "mel_sub_resize_seconds", start)
        window_specs.append(
            {
                "frame_offset": i,
                "window_idx": len(window_specs),
                "frame_key": tuple(frame_sequence[i : i + WINDOW_LEN]),
                "mel_range": (begin, end),
            }
        )

    if not window_specs:
        return []

    top_batch = build_top_mel_tensor_batch(sub_mels, device=preprocess_device, detail=detail)
    windows = build_window_tensors_batched(
        window_specs,
        top_batch,
        frame_tensors,
        label,
        video_path,
        detail=detail,
    )
    if preprocess_device is not None and preprocess_device.type != "cpu":
        start = time.perf_counter()
        sync(preprocess_device)
        add_elapsed(detail, "preprocess_device_sync_seconds", start)
    return windows


def make_batch(windows, device):
    imgs = torch.stack([w.img for w in windows]).to(device, non_blocking=True)
    crops = []
    for crop_group in range(3):
        group = []
        for frame_idx in range(WINDOW_LEN):
            group.append(
                torch.stack([w.crops[crop_group][frame_idx] for w in windows]).to(
                    device, non_blocking=True
                )
            )
        crops.append(group)
    return imgs, crops


@torch.inference_mode()
def infer_pending(model, pending, device, args, scores_rows, timing):
    while len(pending) >= args.batch_size or (args.flush and pending):
        current = pending[: args.batch_size]
        del pending[: args.batch_size]

        start = time.perf_counter()
        imgs, crops = make_batch(current, device)
        sync(device)
        score = model(crops, model.get_features(imgs))[0].sigmoid().flatten()
        sync(device)
        end = time.perf_counter()
        timing["transfer_forward_seconds"] += end - start
        timing["num_batches"] += 1

        score_list = score.cpu().tolist()
        for item, item_score in zip(current, score_list):
            scores_rows.append(
                {
                    "video_path": item.video_path,
                    "window_idx": item.window_idx,
                    "label": item.label,
                    "score": float(item_score),
                }
            )


def run_pipeline(model, tasks, device, args):
    preprocess_device = device if args.preprocess_device == "gpu" else torch.device("cpu")
    detail = {} if args.profile_preprocess_detail else None
    timing = {
        "video_decode_seconds": 0.0,
        "audio_mel_seconds": 0.0,
        "compose_big_image_seconds": 0.0,
        "tensor_transform_seconds": 0.0,
        "transfer_forward_seconds": 0.0,
        "num_batches": 0,
    }
    scores_rows = []
    failures = []
    pending = []
    total_start = time.perf_counter()

    for task in tqdm(tasks, desc="videos"):
        if not task.video_path.exists():
            failures.append({"video_path": str(task.video_path), "error": "missing video"})
            continue
        if not task.audio_path.exists():
            failures.append({"video_path": str(task.video_path), "error": "missing audio"})
            continue

        try:
            start = time.perf_counter()
            frame_count, frame_sequence, frame_list = select_video_frames(
                task.video_path,
                detail=detail,
                num_windows=args.num_windows,
                frame_count_override=task.frame_count,
            )
            timing["video_decode_seconds"] += time.perf_counter() - start

            start = time.perf_counter()
            mel_rgba = official_mel_rgba_in_memory(task.audio_path, detail=detail)
            timing["audio_mel_seconds"] += time.perf_counter() - start

            start = time.perf_counter()
            window_tensors = make_window_tensors(
                frame_count,
                frame_sequence,
                frame_list,
                mel_rgba,
                task.label,
                task.video_path,
                detail=detail,
                preprocess_device=preprocess_device,
            )
            sync(preprocess_device)
            pending.extend(window_tensors)
            timing["tensor_transform_seconds"] += time.perf_counter() - start

            args.flush = False
            infer_pending(model, pending, device, args, scores_rows, timing)
        except Exception as exc:
            failures.append({"video_path": str(task.video_path), "error": repr(exc)})

    args.flush = True
    infer_pending(model, pending, device, args, scores_rows, timing)

    total_seconds = time.perf_counter() - total_start
    num_windows = len(scores_rows)
    num_videos_ok = len({row["video_path"] for row in scores_rows})

    timing.update(
        {
            "total_seconds": total_seconds,
            "num_videos_requested": len(tasks),
            "num_videos_ok": num_videos_ok,
            "num_windows": num_windows,
            "windows_per_second": num_windows / total_seconds if total_seconds else None,
            "videos_per_second": num_videos_ok / total_seconds if total_seconds else None,
            "total_ms_per_window": total_seconds * 1000.0 / num_windows if num_windows else None,
            "transfer_forward_ms_per_window": (
                timing["transfer_forward_seconds"] * 1000.0 / num_windows if num_windows else None
            ),
            "video_decode_ms_per_video": (
                timing["video_decode_seconds"] * 1000.0 / num_videos_ok if num_videos_ok else None
            ),
            "audio_mel_ms_per_video": (
                timing["audio_mel_seconds"] * 1000.0 / num_videos_ok if num_videos_ok else None
            ),
            "pre_model_ms_per_window": (
                (
                    timing["video_decode_seconds"]
                    + timing["audio_mel_seconds"]
                    + timing["compose_big_image_seconds"]
                    + timing["tensor_transform_seconds"]
                )
                * 1000.0
                / num_windows
                if num_windows
                else None
            ),
        }
    )
    if detail is not None:
        timing["preprocess_detail_seconds"] = detail
        timing["preprocess_detail_ms_per_window"] = {
            key.replace("_seconds", "_ms_per_window"): value * 1000.0 / num_windows
            for key, value in detail.items()
        } if num_windows else {}
        timing["preprocess_stage_ms_per_window"] = {
            "video_open_meta": detail.get("video_open_meta_seconds", 0.0) * 1000.0 / num_windows,
            "video_frame_read": detail.get("frame_read_seconds", 0.0) * 1000.0 / num_windows,
            "video_frame_convert_resize": detail.get("frame_convert_resize_seconds", 0.0) * 1000.0 / num_windows,
            "audio_load": detail.get("audio_load_seconds", 0.0) * 1000.0 / num_windows,
            "audio_mel_compute": detail.get("mel_compute_seconds", 0.0) * 1000.0 / num_windows,
            "audio_png_roundtrip": detail.get("mel_png_roundtrip_seconds", 0.0) * 1000.0 / num_windows,
            "mel_window_resize": detail.get("mel_sub_resize_seconds", 0.0) * 1000.0 / num_windows,
            "frame_tensor_cache": detail.get("frame_tensor_cache_seconds", 0.0) * 1000.0 / num_windows,
            "top_mel_tensor_batch": detail.get("top_mel_tensor_batch_seconds", 0.0) * 1000.0 / num_windows,
            "full_image_resize": detail.get("full_image_resize_seconds", 0.0) * 1000.0 / num_windows,
            "crop_source_build": detail.get("crop_source_build_seconds", 0.0) * 1000.0 / num_windows,
            "crop_resize": (
                detail.get("crop0_resize_seconds", 0.0)
                + detail.get("crop_inner_resize_seconds", 0.0)
            ) * 1000.0 / num_windows,
            "preprocess_device_sync": detail.get("preprocess_device_sync_seconds", 0.0) * 1000.0 / num_windows,
        } if num_windows else {}
    return timing, scores_rows, failures


def compute_metrics(scores_rows, threshold):
    labeled = [row for row in scores_rows if row["label"] is not None]
    if not labeled:
        return None

    y_true = np.array([int(row["label"]) for row in labeled])
    y_score = np.array([float(row["score"]) for row in labeled])
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "ap": float(average_precision_score(y_true, y_score)) if len(set(y_true.tolist())) > 1 else None,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def save_scores(path, scores_rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "window_idx", "label", "score"])
        writer.writeheader()
        writer.writerows(scores_rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--clip_name", default=None, choices=["ViT-L/14", "ViT-B/16", "ViT-B/32"])
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--video_root", default="./AVLips")
    parser.add_argument("--audio_root", default="./AVLips/wav")
    parser.add_argument("--video_path", default=None)
    parser.add_argument("--audio_path", default=None)
    parser.add_argument("--label", type=int, choices=[0, 1], default=None)
    parser.add_argument("--max_videos_per_class", type=int, default=5)
    parser.add_argument("--num_windows", type=int, default=N_EXTRACT)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--preprocess_device",
        choices=["cpu", "gpu"],
        default="cpu",
        help=(
            "Where to run batched tensor resize/crop preprocessing. "
            "cpu is the strict default; gpu is experimental and must pass equivalence checks."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--save_scores", default=None)
    parser.add_argument(
        "--profile_preprocess_detail",
        action="store_true",
        help="Record fine-grained preprocessing timings without changing outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, clip_name, backbone = load_model(args, device)
    tasks = discover_tasks(args)
    if not tasks:
        raise SystemExit("No videos found. Check --video_root/--audio_root or use --video_path/--audio_path.")

    print(f"Device: {device}")
    print(f"CLIP: {clip_name}, backbone: {backbone}, global feature dim: {model.global_feature_dim}")
    print(f"Tasks: {len(tasks)}, batch_size: {args.batch_size}")
    print("Intermediate PNG writing: disabled")
    print(f"Batched tensor preprocessing device: {args.preprocess_device}")
    print(f"Detailed preprocessing profile: {args.profile_preprocess_detail}")

    timing, scores_rows, failures = run_pipeline(model, tasks, device, args)
    metrics = compute_metrics(scores_rows, args.threshold)
    result = {
        "ckpt": args.ckpt,
        "clip": clip_name,
        "backbone": backbone,
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "preprocess_device": args.preprocess_device,
        "video_root": args.video_root,
        "audio_root": args.audio_root,
        "video_path": args.video_path,
        "audio_path": args.audio_path,
        "max_videos_per_class": args.max_videos_per_class,
        "device": str(device),
        "params": {
            "total": count_params(model),
            "total_m": count_params(model) / 1e6,
            "encoder_m": count_params(model.encoder) / 1e6,
            "backbone_m": count_params(model.backbone) / 1e6,
        },
        "metrics": metrics,
        "timing": timing,
        "failures": failures,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"saved benchmark: {output}")

    if args.save_scores:
        save_scores(args.save_scores, scores_rows)
        print(f"saved scores: {args.save_scores}")


if __name__ == "__main__":
    main()
