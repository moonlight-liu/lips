import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import HTTPException, UploadFile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "detect_jobs"
WINDOW_LEN = 5
MAX_WINDOWS = 10
MAX_VIDEO_SECONDS = 180.0
MIN_RECOMMENDED_FPS = 10.0

BEST_MODEL = {
    "clip_name": "ViT-B/32",
    "backbone": "resnet18",
    "ckpt": PROJECT_ROOT
    / "lightweight"
    / "results"
    / "checkpoints"
    / "region_resnet18_clip_b32_ra0p01"
    / "best.pth",
}


@dataclass
class VideoMetadata:
    frame_count: int
    fps: float
    width: int
    height: int
    duration_sec: float
    num_windows: int
    warnings: list[str]


async def _save_upload(upload: UploadFile, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with output.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            f.write(chunk)
    await upload.close()
    return size


def _ffmpeg_executable() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="服务器缺少 ffmpeg，无法从视频中提取音频。",
        ) from exc


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    import subprocess

    command = [
        _ffmpeg_executable(),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail="视频音频提取超时。请先裁剪视频，或上传 paired WAV 音频文件。",
        ) from exc

    if completed.returncode != 0 or not audio_path.exists() or audio_path.stat().st_size == 0:
        raise HTTPException(
            status_code=400,
            detail="未能从视频中提取音频轨。请确认视频包含音频，或上传对应的 WAV 文件。",
        )


def _safe_suffix(filename: str | None, default: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if not suffix or len(suffix) > 12:
        return default
    return suffix


def _read_video_metadata(video_path: Path) -> VideoMetadata:
    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="无法读取视频元数据。") from exc

    if frame_count < WINDOW_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"视频过短，至少需要 {WINDOW_LEN} 帧；当前可读帧数为 {frame_count}。",
        )

    duration_sec = frame_count / fps if fps > 0 else 0.0
    num_windows = min(MAX_WINDOWS, frame_count // WINDOW_LEN)
    warnings: list[str] = []

    if fps <= 0:
        warnings.append("无法读取视频 fps，窗口时间仅按序号估算。")
    elif fps < MIN_RECOMMENDED_FPS:
        warnings.append(f"视频 fps 较低（{fps:.2f}），时间轴粒度和分数稳定性可能下降。")

    if height > width:
        warnings.append("检测到竖屏视频；系统会按 LipFD 输入语义 resize，画面比例可能被压缩。")

    if duration_sec > MAX_VIDEO_SECONDS:
        warnings.append(
            f"视频较长（{duration_sec:.1f}s），demo 当前抽取最多 {MAX_WINDOWS} 个窗口进行快速检测。"
        )

    return VideoMetadata(
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        duration_sec=duration_sec,
        num_windows=num_windows,
        warnings=warnings,
    )


def _window_start_times(metadata: VideoMetadata) -> list[float]:
    if metadata.num_windows <= 0:
        return []
    if metadata.fps <= 0:
        return [round(index * 0.5, 2) for index in range(metadata.num_windows)]

    import numpy as np

    starts = np.linspace(
        0,
        metadata.frame_count - WINDOW_LEN - 1,
        metadata.num_windows,
        endpoint=True,
        dtype=np.int32,
    ).tolist()
    return [round(frame / metadata.fps, 2) for frame in starts]


def _scores_to_timeline(scores_rows: list[dict[str, Any]], metadata: VideoMetadata) -> list[dict[str, Any]]:
    start_times = _window_start_times(metadata)
    timeline = []
    for row in sorted(scores_rows, key=lambda item: int(item["window_idx"])):
        score = float(row["score"])
        window_idx = int(row["window_idx"])
        start_time = start_times[window_idx] if window_idx < len(start_times) else round(window_idx * 0.5, 2)
        duration = round(WINDOW_LEN / metadata.fps, 2) if metadata.fps > 0 else 0.0
        timeline.append(
            {
                "index": window_idx,
                "time": start_time,
                "startTime": start_time,
                "endTime": round(start_time + duration, 2),
                "score": round(score, 4),
                "label": "fake" if score >= 0.5 else "real",
            }
        )
    return timeline


def _summarize_result(
    *,
    timing: dict[str, Any],
    timeline: list[dict[str, Any]],
    filename: str | None,
    model_id: str,
    file_size_mb: float,
    elapsed_ms: float,
    metadata: VideoMetadata,
    request_timing: dict[str, float],
) -> dict[str, Any]:
    if not timeline:
        raise HTTPException(status_code=500, detail="模型没有生成任何检测窗口。")

    fake_probability = sum(item["score"] for item in timeline) / len(timeline)
    label = "fake" if fake_probability >= 0.5 else "real"
    latency_ms = timing.get("total_ms_per_window") or elapsed_ms
    fps = timing.get("windows_per_second") or (1000.0 / latency_ms if latency_ms else 0.0)

    return {
        "mode": "real",
        "filename": filename,
        "modelId": model_id,
        "label": label,
        "fakeProbability": round(fake_probability, 4),
        "confidence": round(abs(fake_probability - 0.5) * 2, 4),
        "metrics": {
            "latencyMs": round(float(latency_ms), 2),
            "fps": round(float(fps), 2),
            "windows": len(timeline),
            "fileSizeMb": round(file_size_mb, 2),
        },
        "timeline": timeline,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "warnings": metadata.warnings,
        "video": {
            "frameCount": metadata.frame_count,
            "fps": round(metadata.fps, 3),
            "width": metadata.width,
            "height": metadata.height,
            "durationSec": round(metadata.duration_sec, 3),
            "sampledWindows": metadata.num_windows,
        },
        "backend": {
            "residentModel": True,
            "preprocessDevice": "gpu",
            "clip": BEST_MODEL["clip_name"],
            "backbone": BEST_MODEL["backbone"],
            "totalSeconds": timing.get("total_seconds"),
            "preModelMsPerWindow": timing.get("pre_model_ms_per_window"),
            "transferForwardMsPerWindow": timing.get("transfer_forward_ms_per_window"),
            "videoDecodeMsPerVideo": timing.get("video_decode_ms_per_video"),
            "audioMelMsPerVideo": timing.get("audio_mel_ms_per_video"),
            "preprocessStageMsPerWindow": timing.get("preprocess_stage_ms_per_window"),
            "preprocessDetailMsPerWindow": timing.get("preprocess_detail_ms_per_window"),
            "uploadSaveMs": request_timing.get("upload_save_ms"),
            "audioExtractMs": request_timing.get("audio_extract_ms"),
            "detectElapsedMs": request_timing.get("detect_elapsed_ms"),
            "requestElapsedMs": elapsed_ms,
        },
    }


class ResidentLipFDDetector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._warmed_up = False
        self._warmup_error: str | None = None
        self._warmup_timing: dict[str, Any] | None = None
        self._model = None
        self._device = None
        self._pipeline = None

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            if not BEST_MODEL["ckpt"].exists():
                raise RuntimeError(f"Missing checkpoint: {BEST_MODEL['ckpt']}")

            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            import torch
            from lightweight.scripts import benchmark_video_pipeline as pipeline

            args = SimpleNamespace(
                ckpt=str(BEST_MODEL["ckpt"]),
                clip_name=BEST_MODEL["clip_name"],
                backbone=BEST_MODEL["backbone"],
                gpu=0,
            )
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._model, _, _ = pipeline.load_model(args, self._device)
            self._pipeline = pipeline
            self._loaded = True

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self._loaded,
            "warmedUp": self._warmed_up,
            "warmupError": self._warmup_error,
            "warmupTiming": self._warmup_timing,
            "clip": BEST_MODEL["clip_name"],
            "backbone": BEST_MODEL["backbone"],
            "checkpoint": str(BEST_MODEL["ckpt"]),
            "device": str(self._device) if self._device is not None else None,
        }

    def warm_up(self) -> None:
        self.load()
        if self._warmed_up:
            return

        sample_video = PROJECT_ROOT / "AVLips" / "0_real" / "0.mp4"
        sample_audio = PROJECT_ROOT / "AVLips" / "wav" / "0_real" / "0.wav"
        if not sample_video.exists() or not sample_audio.exists():
            self._warmup_error = "warm-up sample not found"
            return

        try:
            metadata = _read_video_metadata(sample_video)
            metadata.num_windows = 1
            metadata.warnings = []
            start = time.perf_counter()
            timing, scores_rows, failures = self.detect(sample_video, sample_audio, metadata)
            elapsed = time.perf_counter() - start
            if failures:
                self._warmup_error = repr(failures[0])
                return
            self._warmup_timing = {
                "seconds": elapsed,
                "windows": len(scores_rows),
                "totalMsPerWindow": timing.get("total_ms_per_window"),
                "preModelMsPerWindow": timing.get("pre_model_ms_per_window"),
                "transferForwardMsPerWindow": timing.get("transfer_forward_ms_per_window"),
            }
            self._warmup_error = None
            self._warmed_up = True
        except Exception as exc:
            self._warmup_error = repr(exc)

    def detect(self, video_path: Path, audio_path: Path, metadata: VideoMetadata) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        self.load()
        assert self._pipeline is not None
        assert self._model is not None
        assert self._device is not None

        args = SimpleNamespace(
            preprocess_device="gpu" if str(self._device).startswith("cuda") else "cpu",
            profile_preprocess_detail=True,
            batch_size=16,
            num_windows=metadata.num_windows,
        )
        task = self._pipeline.VideoTask(video_path, audio_path, None, metadata.frame_count)
        with self._lock:
            return self._pipeline.run_pipeline(self._model, [task], self._device, args)


_DETECTOR = ResidentLipFDDetector()


def warm_up_detector() -> None:
    _DETECTOR.warm_up()


def detector_status() -> dict[str, Any]:
    return _DETECTOR.status()


async def detect_video(
    file: UploadFile,
    model_id: str,
    audio_file: UploadFile | None = None,
) -> dict[str, Any]:
    if model_id not in {"lipfd-light-best", "region-resnet18"}:
        raise HTTPException(status_code=400, detail="当前真实检测只接入了 LipFD-Light Best 模块。")

    job_id = uuid.uuid4().hex
    job_dir = RUNTIME_ROOT / job_id
    video_path = job_dir / f"input{_safe_suffix(file.filename, '.mp4')}"
    audio_path = job_dir / f"audio{_safe_suffix(audio_file.filename if audio_file else None, '.wav')}"

    request_start = time.perf_counter()
    save_start = time.perf_counter()
    video_size = await _save_upload(file, video_path)
    upload_save_ms = (time.perf_counter() - save_start) * 1000.0
    metadata = _read_video_metadata(video_path)

    if audio_file is None:
        extract_start = time.perf_counter()
        _extract_audio(video_path, audio_path)
        audio_extract_ms = (time.perf_counter() - extract_start) * 1000.0
        audio_size = audio_path.stat().st_size
    else:
        save_start = time.perf_counter()
        audio_size = await _save_upload(audio_file, audio_path)
        upload_save_ms += (time.perf_counter() - save_start) * 1000.0
        audio_extract_ms = 0.0

    start = time.perf_counter()
    try:
        timing, scores_rows, failures = _DETECTOR.detect(video_path, audio_path, metadata)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LipFD 推理失败：{exc}") from exc
    finally:
        detect_elapsed_ms = (time.perf_counter() - start) * 1000.0
        elapsed_ms = (time.perf_counter() - request_start) * 1000.0

    if failures:
        raise HTTPException(status_code=500, detail=f"LipFD pipeline failure: {failures[0]}")

    timeline = _scores_to_timeline(scores_rows, metadata)
    result = _summarize_result(
        timing=timing,
        timeline=timeline,
        filename=file.filename,
        model_id=model_id,
        file_size_mb=(video_size + audio_size) / (1024 * 1024),
        elapsed_ms=elapsed_ms,
        metadata=metadata,
        request_timing={
            "detect_elapsed_ms": detect_elapsed_ms,
            "upload_save_ms": upload_save_ms,
            "audio_extract_ms": audio_extract_ms,
        },
    )

    shutil.rmtree(job_dir, ignore_errors=True)
    return result
