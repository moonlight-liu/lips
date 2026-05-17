import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import feature as audio
from tqdm import tqdm


"""
Structure of the AVLips dataset:
AVLips
├── 0_real
├── 1_fake
└── wav
    ├── 0_real
    └── 1_fake
"""

N_EXTRACT = 10
WINDOW_LEN = 5
audio_root = "./AVLips/wav"
video_root = "./AVLips"
output_root = "./datasets/AVLips"
labels = [(0, "0_real"), (1, "1_fake")]


def get_spectrogram(audio_file):
    data, sr = librosa.load(audio_file, sr=None)
    mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)
    mel = mel.astype(np.float32)
    mel = (mel - mel.min()) / max(float(mel.max() - mel.min()), 1e-6)
    return plt.get_cmap("viridis")(mel, bytes=True)[:, :, :3]


def select_frame_indices(frame_count):
    max_start = frame_count - WINDOW_LEN - 1
    if max_start < 0:
        return []
    return np.linspace(0, max_start, N_EXTRACT, endpoint=True, dtype=np.int32).tolist()


def process_video(task):
    dataset_name, video_name, skip_existing, png_compression = task
    root = os.path.join(video_root, dataset_name)
    video_path = os.path.join(root, video_name)
    name = os.path.splitext(video_name)[0]
    audio_path = os.path.join(audio_root, dataset_name, f"{name}.wav")
    output_dir = os.path.join(output_root, dataset_name)
    expected_outputs = [
        os.path.join(output_dir, f"{name}_{group}.png")
        for group in range(N_EXTRACT)
    ]

    if skip_existing and all(os.path.exists(p) for p in expected_outputs):
        return video_name, 0, None

    if not os.path.exists(audio_path):
        return video_name, 0, f"missing audio: {audio_path}"

    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = select_frame_indices(frame_count)
    if not frame_idx:
        video_capture.release()
        return video_name, 0, f"too few frames: {frame_count}"

    frame_sequence = [i for num in frame_idx for i in range(num, num + WINDOW_LEN)]
    needed = set(frame_sequence)
    frame_list = []
    current_frame = 0
    while current_frame <= frame_sequence[-1]:
        ret, frame = video_capture.read()
        if not ret:
            video_capture.release()
            return video_name, 0, f"failed at frame {current_frame}"
        if current_frame in needed:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(cv2.resize(frame, (500, 500), interpolation=cv2.INTER_AREA))
        current_frame += 1
    video_capture.release()

    mel = get_spectrogram(audio_path)
    mapping = mel.shape[1] / frame_count
    saved = 0
    for i in range(0, len(frame_list), WINDOW_LEN):
        begin = int(np.round(frame_sequence[i] * mapping))
        end = int(np.round((frame_sequence[i] + WINDOW_LEN) * mapping))
        if end <= begin:
            continue
        sub_mel = cv2.resize(mel[:, begin:end], (500 * WINDOW_LEN, 500), interpolation=cv2.INTER_AREA)
        frames = np.concatenate(frame_list[i : i + WINDOW_LEN], axis=1)
        if frames.shape[1] != 500 * WINDOW_LEN:
            continue
        sample = np.concatenate((sub_mel[:, :, :3], frames[:, :, :3]), axis=0)
        out_path = os.path.join(output_dir, f"{name}_{saved}.png")
        cv2.imwrite(
            out_path,
            cv2.cvtColor(sample, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, png_compression],
        )
        saved += 1

    return video_name, saved, None


def run(max_sample=None, workers=1, skip_existing=True, png_compression=1):
    for _, dataset_name in labels:
        output_dir = os.path.join(output_root, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        root = os.path.join(video_root, dataset_name)
        video_list = sorted(v for v in os.listdir(root) if v.lower().endswith(".mp4"))
        if max_sample is not None:
            video_list = video_list[:max_sample]

        print(f"Handling {dataset_name}: {len(video_list)} videos")
        tasks = [(dataset_name, v, skip_existing, png_compression) for v in video_list]
        failures = []

        if workers <= 1:
            iterator = (process_video(task) for task in tasks)
            for video_name, saved, error in tqdm(iterator, total=len(tasks)):
                if error:
                    failures.append((video_name, error))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(process_video, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    video_name, saved, error = future.result()
                    if error:
                        failures.append((video_name, error))

        if failures:
            print(f"{dataset_name}: {len(failures)} failures")
            for video_name, error in failures[:20]:
                print(f"  {video_name}: {error}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_sample", type=int, default=None, help="limit videos per class for a quick test")
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) // 2, 1))
    parser.add_argument("--no_skip_existing", action="store_true", help="rewrite samples even when all expected images exist")
    parser.add_argument("--png_compression", type=int, default=1, choices=range(10), metavar="[0-9]")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(output_root, exist_ok=True)
    run(
        max_sample=args.max_sample,
        workers=args.workers,
        skip_existing=not args.no_skip_existing,
        png_compression=args.png_compression,
    )
