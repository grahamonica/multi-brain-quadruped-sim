"""Temporary standalone command-primitives demo server.

Run with:
    uvicorn brains.api.command_demo:app --host 127.0.0.1 --port 8010
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from brains.config import DEFAULT_CONFIG_PATH, load_runtime_spec
from brains.harnesses.direction_harness import DirectionHarness

from .common import BroadcastHub, build_tracking_camera, encode_rgb_frame


COMMANDS: tuple[str, ...] = (
    "stand",
    "walk",
    "trot",
    "skip",
    "turn_left",
    "turn_right",
    "front_flip",
    "back_flip",
    "side_roll",
)
DEFAULT_COMMAND = "stand"
DEFAULT_SPEED = 0.65
COMMAND_DEFAULT_SPEEDS: dict[str, float] = {
    "stand": 0.0,
    "walk": 0.40,
    "trot": 0.50,
    "skip": 0.45,
    "turn_left": 0.55,
    "turn_right": 0.55,
    "front_flip": 1.00,
    "back_flip": 1.00,
    "side_roll": 0.75,
}
VIEWER_FRAME_WIDTH = int(os.environ.get("QUADRUPED_VIEWER_WIDTH", "640"))
VIEWER_FRAME_HEIGHT = int(os.environ.get("QUADRUPED_VIEWER_HEIGHT", "360"))
VIEWER_CAMERA_DISTANCE_M = float(os.environ.get("QUADRUPED_VIEWER_CAMERA_DISTANCE_M", "2.8"))
VIEWER_CAMERA_AZIMUTH_DEG = float(os.environ.get("QUADRUPED_VIEWER_CAMERA_AZIMUTH_DEG", "132.0"))
VIEWER_CAMERA_ELEVATION_DEG = float(os.environ.get("QUADRUPED_VIEWER_CAMERA_ELEVATION_DEG", "-20.0"))


@dataclass(frozen=True)
class DemoSnapshot:
    command: str
    speed: float
    version: int


class DemoControls:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._command = DEFAULT_COMMAND
        self._speed = DEFAULT_SPEED
        self._version = 0

    def snapshot(self) -> DemoSnapshot:
        with self._lock:
            return DemoSnapshot(command=self._command, speed=self._speed, version=self._version)

    def set_command(self, command: str, speed: float | None = None) -> DemoSnapshot:
        with self._lock:
            self._command = command
            if speed is not None:
                self._speed = float(np.clip(speed, 0.0, 1.0))
            self._version += 1
            return DemoSnapshot(command=self._command, speed=self._speed, version=self._version)


def _load_config_path() -> Path:
    return Path(os.environ.get("QUADRUPED_CONFIG", str(DEFAULT_CONFIG_PATH)))


def _arg_value(argv: list[str], flag: str) -> str | None:
    for index, token in enumerate(argv):
        if token == flag and (index + 1) < len(argv):
            return argv[index + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _listener_pids(port: int) -> list[int]:
    if port <= 0:
        return []
    result = subprocess.run(
        ["lsof", "-nP", "-tiTCP:{port}".format(port=port), "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
    )
    pids: list[int] = []
    for raw_pid in result.stdout.split():
        try:
            pids.append(int(raw_pid))
        except ValueError:
            continue
    return pids


def _kill_port_listeners(port: int) -> None:
    if port <= 0:
        return
    current_pid = os.getpid()
    target_pids = [pid for pid in _listener_pids(port) if pid != current_pid]
    for pid in target_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue

    deadline = time.time() + 3.0
    while time.time() < deadline:
        remaining = [pid for pid in _listener_pids(port) if pid != current_pid]
        if not remaining:
            return
        time.sleep(0.05)

    for pid in [pid for pid in _listener_pids(port) if pid != current_pid]:
        if pid == current_pid:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            continue

    # Give the kernel a short moment to release the socket after SIGKILL.
    time.sleep(0.1)


def _autokill_previous_demo_server() -> None:
    if os.environ.get("QUADRUPED_AUTOKILL_DEMO_PORT", "1").strip().lower() in {"0", "false", "off", "no"}:
        return
    argv = list(sys.argv)
    if len(argv) < 2:
        return
    entrypoint = Path(argv[0]).name.lower()
    if "uvicorn" not in entrypoint and "uvicorn" not in argv[0].lower():
        return
    if "brains.api.command_demo:app" not in argv:
        return
    port_token = _arg_value(argv, "--port")
    try:
        port = int(port_token) if port_token is not None else 8010
    except (TypeError, ValueError):
        port = 8010
    _kill_port_listeners(port)


_autokill_previous_demo_server()


def _demo_loop(hub: BroadcastHub, controls: DemoControls, config_path: Path) -> None:
    import mujoco

    spec = load_runtime_spec(config_path)
    spec = replace(
        spec,
        control=replace(
            spec.control,
            mode="motor_targets",
        ),
    )
    spec.validate()

    harness = DirectionHarness(spec)
    backend = harness._make_backend()
    goal_xyz = harness.default_goal()
    spawn_xy = np.zeros((2,), dtype=np.float32)

    render_model = backend.model
    render_camera = build_tracking_camera(
        mujoco,
        backend._torso_body_id,
        distance_m=VIEWER_CAMERA_DISTANCE_M,
        azimuth_deg=VIEWER_CAMERA_AZIMUTH_DEG,
        elevation_deg=VIEWER_CAMERA_ELEVATION_DEG,
    )
    frame_width = min(VIEWER_FRAME_WIDTH, 640)
    frame_height = min(VIEWER_FRAME_HEIGHT, 360)
    renderer = mujoco.Renderer(render_model, width=frame_width, height=frame_height)

    stream_id: str | None = None
    active_version = -1
    step_index = 0
    data = backend.reset_data(spawn_xy=spawn_xy)
    metrics = backend.initial_metrics(data, goal_xyz)

    hub.publish(
        {
            "type": "metadata",
            "mode": "command_demo",
            "commands": list(COMMANDS),
            "brain_dt_s": float(spec.episode.brain_dt_s),
            "frame_width": frame_width,
            "frame_height": frame_height,
        }
    )

    try:
        while True:
            snapshot = controls.snapshot()
            if snapshot.version != active_version:
                active_version = snapshot.version
                step_index = 0
                data = backend.reset_data(spawn_xy=spawn_xy)
                metrics = backend.initial_metrics(data, goal_xyz)
                stream_id = f"command:{snapshot.command}:{snapshot.version}:{time.time_ns()}"
                hub.publish(
                    {
                        "type": "command_state",
                        "command": snapshot.command,
                        "speed": snapshot.speed,
                        "version": snapshot.version,
                        "stream_id": stream_id,
                    }
                )

            time_s = step_index * float(spec.episode.brain_dt_s)
            target_velocity = harness.target_velocity(snapshot.command, time_s, speed=snapshot.speed)
            backend._advance(data, target_velocity)
            metrics = backend._step_metrics(data, metrics, goal_xyz)
            step_message = backend._snapshot(
                data,
                metrics,
                goal_xyz,
                step_index,
                2_000_000_000,
                policy_output=target_velocity,
                selected_command=snapshot.command,
            )

            renderer.update_scene(data, camera=render_camera)
            rgb = np.asarray(renderer.render(), dtype=np.uint8)

            hub.publish(
                {
                    "type": "frame",
                    "stream_id": stream_id,
                    "step": int(step_index),
                    "time_s": float(step_message.get("time_s", time_s)),
                    "reward": float(step_message.get("reward", 0.0)),
                    "selected_command": snapshot.command,
                    "render": encode_rgb_frame(rgb),
                }
            )

            step_index += 1
            time.sleep(float(spec.episode.brain_dt_s))
    finally:
        renderer.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub = BroadcastHub()
    controls = DemoControls()
    config_path = _load_config_path()

    hub.attach_loop(asyncio.get_running_loop())
    thread = threading.Thread(target=_demo_loop, args=(hub, controls, config_path), daemon=True, name="command-demo")

    app.state.hub = hub
    app.state.controls = controls
    app.state.thread = thread
    thread.start()
    yield


app = FastAPI(title="Quadruped Command Demo", lifespan=lifespan)
_raw_origins = os.environ.get("QUADRUPED_CORS_ORIGINS", "*")
_allowed_origins = [origin.strip() for origin in _raw_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials="*" not in _allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Quadruped Command Demo</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; font-family: Menlo, Consolas, monospace; background: #0b0e11; color: #dbe5ef; }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 14px; }
    .row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
    button { border: 1px solid #2f3f4c; background: #16222b; color: #dbe5ef; padding: 8px 10px; cursor: pointer; }
    button.active { background: #1f5d3a; border-color: #2d8f58; }
    .meta { font-size: 12px; color: #9fb1c3; }
    canvas { width: 100%; max-width: 100%; height: auto; border: 1px solid #21303a; background: black; }
  </style>
</head>
<body>
  <div class="wrap">
    <h3>Temporary Command Demo</h3>
    <div class="row" id="buttons"></div>
    <div class="meta" id="status">connecting...</div>
    <canvas id="view"></canvas>
  </div>
  <script>
    const commands = ["stand", "walk", "trot", "skip", "turn_left", "turn_right", "front_flip", "back_flip", "side_roll"];
    const statusEl = document.getElementById("status");
    const buttonsEl = document.getElementById("buttons");
    const canvas = document.getElementById("view");
    const ctx = canvas.getContext("2d", { alpha: false });
    let active = "stand";

    function drawButtons() {
      buttonsEl.innerHTML = "";
      for (const command of commands) {
        const button = document.createElement("button");
        button.textContent = command;
        if (command === active) button.classList.add("active");
        button.onclick = () => {
          ws.send(JSON.stringify({ type: "set_command", command }));
        };
        buttonsEl.appendChild(button);
      }
    }

    function decodeRgbBase64(base64, expectedLength) {
      const binary = atob(base64);
      const length = Math.min(binary.length, expectedLength);
      const bytes = new Uint8ClampedArray(length);
      for (let i = 0; i < length; i += 1) bytes[i] = binary.charCodeAt(i);
      return bytes;
    }

    function drawFrame(frame) {
      const render = frame.render;
      if (!render) return;
      const width = Number(render.width || 0);
      const height = Number(render.height || 0);
      const channels = Number(render.channels || 3);
      if (width <= 0 || height <= 0) return;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const pixels = decodeRgbBase64(render.rgb, width * height * channels);
      if (channels === 3 && pixels.length === width * height * 3) {
        const rgba = new Uint8ClampedArray(width * height * 4);
        for (let src = 0, dst = 0; src < pixels.length; src += 3, dst += 4) {
          rgba[dst] = pixels[src];
          rgba[dst + 1] = pixels[src + 1];
          rgba[dst + 2] = pixels[src + 2];
          rgba[dst + 3] = 255;
        }
        ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
      }
      statusEl.textContent = `connected | command=${frame.selected_command || active} | step=${frame.step} | t=${Number(frame.time_s || 0).toFixed(2)}s`;
    }

    const wsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
    const ws = new WebSocket(wsUrl);
    ws.onopen = () => { statusEl.textContent = "connected"; };
    ws.onclose = () => { statusEl.textContent = "disconnected"; };
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "command_state") {
        active = String(message.command || active);
        drawButtons();
      } else if (message.type === "frame") {
        drawFrame(message);
      }
    };

    drawButtons();
  </script>
</body>
</html>"""


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    hub: BroadcastHub = websocket.app.state.hub
    controls: DemoControls = websocket.app.state.controls
    await hub.connect(websocket)
    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                continue

            message_type = str(message.get("type", ""))
            if message_type != "set_command":
                continue

            command = str(message.get("command", "")).strip()
            if command not in COMMANDS:
                continue
            speed_raw = message.get("speed")
            if speed_raw is not None:
                speed = float(speed_raw)
            else:
                speed = COMMAND_DEFAULT_SPEEDS.get(command, DEFAULT_SPEED)
            snapshot = controls.set_command(command, speed=speed)
            hub.publish(
                {
                    "type": "command_state",
                    "command": snapshot.command,
                    "speed": snapshot.speed,
                    "version": snapshot.version,
                }
            )
    except WebSocketDisconnect:
        hub.disconnect(websocket)
    except Exception:
        hub.disconnect(websocket)
