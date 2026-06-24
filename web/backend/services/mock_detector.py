import hashlib
import math
import random
from datetime import datetime, timezone
from typing import Any

from fastapi import UploadFile


MODEL_LATENCY = {
    "original": 106.4,
    "region-resnet18": 38.2,
    "region-resnet34": 54.7,
}


async def mock_detect_video(file: UploadFile, model_id: str, mode: str = "mock") -> dict[str, Any]:
    payload = await file.read()
    seed_material = f"{file.filename}:{model_id}:{len(payload)}".encode("utf-8")
    seed = int(hashlib.sha256(seed_material).hexdigest()[:12], 16)
    rng = random.Random(seed)

    base_score = 0.22 + rng.random() * 0.62
    windows = 18 + seed % 12
    latency_ms = MODEL_LATENCY.get(model_id, 42.0) + rng.uniform(-4.5, 5.5)
    latency_ms = max(18.0, latency_ms)
    fps = 1000.0 / latency_ms

    timeline = []
    for idx in range(windows):
        wave = math.sin(idx / 2.8) * 0.08
        noise = rng.uniform(-0.055, 0.055)
        score = min(0.98, max(0.02, base_score + wave + noise))
        timeline.append(
            {
                "index": idx,
                "time": round(idx * 0.5, 2),
                "score": round(score, 4),
                "label": "fake" if score >= 0.5 else "real",
            }
        )

    fake_probability = sum(item["score"] for item in timeline) / len(timeline)
    label = "fake" if fake_probability >= 0.5 else "real"

    return {
        "mode": mode,
        "filename": file.filename,
        "modelId": model_id,
        "label": label,
        "fakeProbability": round(fake_probability, 4),
        "confidence": round(abs(fake_probability - 0.5) * 2, 4),
        "metrics": {
            "latencyMs": round(latency_ms, 2),
            "fps": round(fps, 2),
            "windows": windows,
            "fileSizeMb": round(len(payload) / (1024 * 1024), 2),
        },
        "timeline": timeline,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
