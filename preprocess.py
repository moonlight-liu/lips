import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
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


def get_spectrogram(audio_file, temp_path):
    # Keep official semantics: librosa.load() resamples audio to 22050 Hz by default.
    data, sr = librosa.load(audio_file)
    mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)
    plt.imsave(temp_path, mel)


def process_video(task):
    dataset_name, video_name, temp_dir = task
    root = f"{video_root}/{dataset_name}"
    video_path = f"{root}/{video_name}"
    name = video_name.split(".")[0]
    audio_path = f"{audio_root}/{dataset_name}/{name}.wav"
    output_dir = f"{output_root}/{dataset_name}"
    temp_path = os.path.join(temp_dir, f"mel_{os.getpid()}.png")

    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= WINDOW_LEN:
        video_capture.release()
        return video_name, 0, f"too few frames: {frame_count}"

    # Official code used uint8; int32 keeps the same selected values without overflowing long videos.
    frame_idx = np.linspace(
        0,
        frame_count - WINDOW_LEN - 1,
        N_EXTRACT,
        endpoint=True,
        dtype=np.int32,
    ).tolist()
    frame_idx.sort()
    frame_sequence = [i for num in frame_idx for i in range(num, num + WINDOW_LEN)]

    frame_list = []
    current_frame = 0
    while current_frame <= frame_sequence[-1]:
        ret, frame = video_capture.read()
        if not ret:
            video_capture.release()
            return video_name, 0, f"failed reading frame {current_frame}"
        if current_frame in frame_sequence:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame_list.append(cv2.resize(frame, (500, 500)))
        current_frame += 1
    video_capture.release()

    if not os.path.exists(audio_path):
        return video_name, 0, f"missing audio: {audio_path}"

    group = 0
    get_spectrogram(audio_path, temp_path)
    mel = plt.imread(temp_path) * 255
    mel = mel.astype(np.uint8)
    mapping = mel.shape[1] / frame_count

    for i in range(len(frame_list)):
        idx = i % WINDOW_LEN
        if idx == 0:
            try:
                begin = np.round(frame_sequence[i] * mapping)
                end = np.round((frame_sequence[i] + WINDOW_LEN) * mapping)
                sub_mel = cv2.resize(
                    mel[:, int(begin) : int(end)], (500 * WINDOW_LEN, 500)
                )
                x = np.concatenate(frame_list[i : i + WINDOW_LEN], axis=1)
                x = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)
                plt.imsave(f"{output_dir}/{name}_{group}.png", x)
                group += 1
            except ValueError:
                return video_name, group, f"ValueError: {name}"

    return video_name, group, None


def run(max_sample=None, workers=1, temp_dir="./temp"):
    os.makedirs(temp_dir, exist_ok=True)
    for _, dataset_name in labels:
        os.makedirs(f"{output_root}/{dataset_name}", exist_ok=True)
        root = f"{video_root}/{dataset_name}"
        video_list = sorted(v for v in os.listdir(root) if v.lower().endswith(".mp4"))
        if max_sample is not None:
            video_list = video_list[:max_sample]

        print(f"Handling {dataset_name}: {len(video_list)} videos")
        tasks = [(dataset_name, v, temp_dir) for v in video_list]
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
    parser.add_argument("--max_sample", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--temp_dir", default="./temp")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(output_root, exist_ok=True)
    run(max_sample=args.max_sample, workers=args.workers, temp_dir=args.temp_dir)
