from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from services.detector import detect_video, detector_status, warm_up_detector
from services.mock_detector import mock_detect_video


app = FastAPI(
    title="LipFD Web Demo API",
    description="FastAPI backend for LipFD real-time detection demo.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    warm_up_detector()


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "lipfd-web-backend", "detector": detector_status()}


@app.get("/api/detector/status")
def status():
    return detector_status()


@app.get("/api/models")
def models():
    return {
        "models": [
            {
                "id": "lipfd-light-best",
                "name": "LipFD-Light Best",
                "description": "ViT-B/32 + ResNet18, GPU batch resize/crop.",
                "status": "ready",
            },
            {
                "id": "region-resnet18",
                "name": "Lightweight Region ResNet18",
                "description": "Alias of the current best LipFD-Light module.",
                "status": "ready",
            },
            {
                "id": "original",
                "name": "Original LipFD",
                "description": "ViT-L/14 + ResNet50 baseline.",
                "status": "placeholder",
            },
        ]
    }


@app.post("/api/detect/mock")
async def detect_mock(model_id: str = "region-resnet18", file: UploadFile = File(...)):
    return await mock_detect_video(file=file, model_id=model_id)


@app.post("/api/detect")
async def detect(
    model_id: str = "lipfd-light-best",
    file: UploadFile = File(...),
    audio_file: UploadFile | None = File(None),
):
    return await detect_video(file=file, audio_file=audio_file, model_id=model_id)
