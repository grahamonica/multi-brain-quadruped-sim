# Frontend

React + Vite 3D viewer for the quadruped websocket streams.

## What It Does

- connects to `ws://localhost:8000/ws` locally or `VITE_WS_URL` in hosted builds
- renders the arena and robot in interactive 3D from backend metadata
- renders selected-model replay frames from the viewer service
- queues frame batches locally so playback stays smooth
- lets the user orbit/pan/zoom the camera, select saved model artifacts, and place reward targets by clicking the scene
- shows generation metrics, checkpoint metadata, and buffer status

## Run

```bash
npm install
npm run dev -- --port 5173
```

The viewer app expects the backend to already be running. In normal use, start both together from the repo root:

```bash
python3 main.py
```

## Notes

- The viewer app no longer hardcodes the active terrain and robot dimensions as the source of truth. It uses websocket metadata emitted by the backend.
- Production builds can be verified with `npm run build`.

## Production Build

```bash
VITE_WS_URL=wss://api.your-domain.example/ws npm run build
```

Deploy `dist/` to a static host. Run the FastAPI backend separately behind HTTPS/WSS:

```bash
QUADRUPED_CONFIG=configs/default.yaml \
QUADRUPED_CHECKPOINT_ROOT=checkpoints \
QUADRUPED_CORS_ORIGINS=https://your-domain.example \
uvicorn brains.api.live:app --host 0.0.0.0 --port 8000
```
