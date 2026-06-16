#!/usr/bin/env python
"""Check official PNG preprocessing vs in-memory video preprocessing.

The official branch writes one temporary 1000x2500 PNG and reads it through
data/datasets.py. The in-memory branch skips the PNG write/read and directly
creates the same tensors. This script compares tensor shapes, max/mean absolute
differences, and optional model score differences.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_video_pipeline import (  # noqa: E402
    load_model,
    make_big_images,
    make_window_tensors,
    official_mel_rgba_in_memory,
    select_video_frames,
)


def build_official_dataset_tensors(png_path, label):
    from data.datasets import AVLip

    root = png_path.parent.parent
    opt = argparse.Namespace(
        data_label="val",
        real_list_path=str(root / "0_real"),
        fake_list_path=str(root / "1_fake"),
    )
    dataset = AVLip(opt)
    if len(dataset) != 1:
        raise RuntimeError(f"Expected exactly 1 sample in temp dataset, got {len(dataset)}")
    img, crops, got_label = dataset[0]
    if int(got_label) != int(label):
        raise RuntimeError(f"Label mismatch: expected {label}, got {got_label}")
    return img, crops


def big_rgb_to_dataset_tensors(big_rgb):
    """Mimic the current server data/datasets.py exactly.

    Current data/datasets.py reads with cv2.imread, keeps BGR channel order and
    0-255 float range, and applies torchvision Resize directly to tensors.
    """
    bgr = big_rgb[:, :, ::-1].copy()
    img = torch.tensor(bgr, dtype=torch.float32)
    img = img.permute(2, 0, 1)

    resize_224 = transforms.Resize((224, 224))
    crops = [[resize_224(img[:, 500:, i : i + 500]) for i in range(5)], [], []]
    crop_idx = [(28, 196), (61, 163)]
    for i in range(len(crops[0])):
        crops[1].append(
            resize_224(
                crops[0][i][
                    :,
                    crop_idx[0][0] : crop_idx[0][1],
                    crop_idx[0][0] : crop_idx[0][1],
                ]
            )
        )
        crops[2].append(
            resize_224(
                crops[0][i][
                    :,
                    crop_idx[1][0] : crop_idx[1][1],
                    crop_idx[1][0] : crop_idx[1][1],
                ]
            )
        )
    img = transforms.Resize((1120, 1120))(img)
    return img, crops


def prepare_temp_png(video_path, audio_path, label, window_index, temp_root):
    frame_count, frame_sequence, frame_list = select_video_frames(video_path)
    mel_rgba = official_mel_rgba_in_memory(audio_path)
    big_images = make_big_images(frame_count, frame_sequence, frame_list, mel_rgba)
    if not big_images:
        raise RuntimeError("No windows produced from video")
    if window_index < 0 or window_index >= len(big_images):
        raise ValueError(f"--window_index must be in [0, {len(big_images) - 1}], got {window_index}")

    group, big_rgb = big_images[window_index]
    class_dir = temp_root / ("1_fake" if label == 1 else "0_real")
    other_dir = temp_root / ("0_real" if label == 1 else "1_fake")
    class_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)
    png_path = class_dir / f"{video_path.stem}_{group}.png"

    # Match official preprocess.py: plt.imsave(output_png, x)
    plt.imsave(png_path, big_rgb)
    return png_path, group, big_rgb, len(big_images), frame_count, frame_sequence, frame_list, mel_rgba


def compare_tensors(name, official, memory):
    diff = (official - memory).abs()
    return {
        "name": name,
        "official_shape": list(official.shape),
        "memory_shape": list(memory.shape),
        "shape_equal": list(official.shape) == list(memory.shape),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def compare_all(official_img, official_crops, memory_img, memory_crops):
    results = [compare_tensors("img", official_img, memory_img)]
    for group_idx in range(len(official_crops)):
        for frame_idx in range(len(official_crops[group_idx])):
            results.append(
                compare_tensors(
                    f"crops[{group_idx}][{frame_idx}]",
                    official_crops[group_idx][frame_idx],
                    memory_crops[group_idx][frame_idx],
                )
            )
    return results


def crops_to_cpu(crops):
    return [[frame.detach().cpu() for frame in crop_group] for crop_group in crops]


def batch_one(img, crops, device):
    batched_img = img.unsqueeze(0).to(device)
    batched_crops = [
        [frame.unsqueeze(0).to(device) for frame in crop_group]
        for crop_group in crops
    ]
    return batched_img, batched_crops


@torch.inference_mode()
def score_one(model, img, crops, device):
    batched_img, batched_crops = batch_one(img, crops, device)
    score = model(batched_crops, model.get_features(batched_img))[0].sigmoid().flatten()[0]
    return float(score.item())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--label", type=int, choices=[0, 1], required=True)
    parser.add_argument("--window_index", type=int, default=0)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--clip_name", default=None, choices=["ViT-L/14", "ViT-B/16", "ViT-B/32"])
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--preprocess_device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Check CPU preprocessing or experimental GPU batch resize/crop preprocessing.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--keep_temp_png", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video_path)
    audio_path = Path(args.audio_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    temp_dir = Path(tempfile.mkdtemp(prefix="lipfd_equiv_"))
    try:
        (
            png_path,
            group,
            big_rgb,
            num_windows,
            frame_count,
            frame_sequence,
            frame_list,
            mel_rgba,
        ) = prepare_temp_png(
            video_path, audio_path, args.label, args.window_index, temp_dir
        )
        official_img, official_crops = build_official_dataset_tensors(png_path, args.label)
        preprocess_device = torch.device(
            f"cuda:{args.gpu}"
            if args.preprocess_device == "gpu" and torch.cuda.is_available()
            else "cpu"
        )
        direct_windows = make_window_tensors(
            frame_count,
            frame_sequence,
            frame_list,
            mel_rgba,
            args.label,
            video_path,
            preprocess_device=preprocess_device,
        )
        memory_window = direct_windows[args.window_index]
        memory_img = memory_window.img.detach().cpu()
        memory_crops = crops_to_cpu(memory_window.crops)
        comparisons = compare_all(official_img, official_crops, memory_img, memory_crops)

        max_abs_diff = max(item["max_abs_diff"] for item in comparisons)
        mean_abs_diff = sum(item["mean_abs_diff"] for item in comparisons) / len(comparisons)
        shape_all_equal = all(item["shape_equal"] for item in comparisons)

        score_result = None
        if args.ckpt:
            device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
            model, clip_name, backbone = load_model(args, device)
            official_score = score_one(model, official_img, official_crops, device)
            memory_score = score_one(model, memory_img, memory_crops, device)
            score_result = {
                "clip": clip_name,
                "backbone": backbone,
                "official_score": official_score,
                "memory_score": memory_score,
                "abs_diff": abs(official_score - memory_score),
            }

        result = {
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "label": args.label,
            "window_index": args.window_index,
            "official_window_group": group,
            "num_windows": num_windows,
            "temp_png": str(png_path) if args.keep_temp_png else None,
            "big_image_shape": list(big_rgb.shape),
            "preprocess_device": args.preprocess_device,
            "summary": {
                "shape_all_equal": shape_all_equal,
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff_across_tensors": mean_abs_diff,
            },
            "comparisons": comparisons,
            "score": score_result,
        }

        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"saved result: {output}")
    finally:
        if args.keep_temp_png:
            print(f"kept temp dir: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
