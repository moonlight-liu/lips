# LipFD Web Demo

Lightweight front-end and back-end shell for the LipFD real-time detection demo.

## Structure

```text
web/
  backend/   FastAPI service with mock detector and real detector hook
  frontend/  Vue 3 + Vite + TypeScript interface
```

## Run Backend

```bash
cd /root/lx/LipFD/web/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Run Frontend

```bash
cd /root/lx/LipFD/web/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

## Local Preview Through SSH

```powershell
ssh -p 12123 -L 5173:127.0.0.1:5173 -L 8000:127.0.0.1:8000 root@172.28.7.26
```

Then open:

```text
http://127.0.0.1:5173
```

## Current Scope

The first version uses `/api/detect/mock` to keep the user interface and API stable while model training is still running. Replace `backend/services/detector.py` later to connect real LipFD preprocessing and inference.
