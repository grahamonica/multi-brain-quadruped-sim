import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_PORT = import.meta.env.VITE_API_PORT || "8000";
const WS_URL =
  import.meta.env.VITE_WS_URL ||
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.hostname}:${API_PORT}/ws`;

const DEFAULT_METADATA = {
  mode: "viewer",
  config_name: "default",
  terrain: {
    kind: "stepped_arena",
    field_half_m: 6.0,
    center_half_m: 2.5,
    step_count: 5,
    step_width_m: 2.0,
    step_height_m: 0.15,
    floor_height_m: 0.0,
  },
  robot: {
    body_length_m: 0.28,
    body_width_m: 0.12,
    body_height_m: 0.08,
    leg_length_m: 0.16,
    leg_radius_m: 0.02,
    foot_radius_m: 0.03,
  },
  model: {
    active: "shared_trunk_es",
    architecture: "shared_trunk_motor_lanes",
    trainer: "openai_es",
    registered: [],
  },
  goal: {
    strategy: "radial_random",
    radius_m: 3.0,
    height_m: 0.16,
    fixed_goal_xyz: null,
  },
  training: {
    population_size: 32,
    episode_s: 30.0,
    selection_interval_s: 15.0,
    viewer_reset_s: 30.0,
    brain_dt_s: 0.05,
  },
  simulator: {
    backend: "unified",
  },
};

const STEP_COLORS = ["#1f7a5f", "#317f95", "#5969a6", "#a66b43", "#b84f62", "#8654a3"];
const TRAIL_LIMIT = 180;
const FRAME_BUFFER_LIMIT = 900;

function basename(value) {
  if (!value) return "uninitialized";
  const parts = String(value).split("/");
  return parts[parts.length - 1] || value;
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "0.00";
  return Number(value).toFixed(digits);
}

function terrainHalf(terrain) {
  return Math.max(terrain?.field_half_m || 6.0, 1.0);
}

function mapTransform(width, height, terrain) {
  const half = terrainHalf(terrain);
  const padding = Math.max(34, Math.min(width, height) * 0.06);
  const scale = Math.min((width - padding * 2) / (half * 2), (height - padding * 2) / (half * 2));
  return {
    cx: width / 2,
    cy: height / 2,
    scale,
    half,
  };
}

function worldToScreen(transform, x, y) {
  return [transform.cx + x * transform.scale, transform.cy - y * transform.scale];
}

function screenToWorld(transform, x, y) {
  return [(x - transform.cx) / transform.scale, (transform.cy - y) / transform.scale];
}

function drawSquare(ctx, transform, half, fill, stroke, width = 1) {
  const [x0, y0] = worldToScreen(transform, -half, half);
  const [x1, y1] = worldToScreen(transform, half, -half);
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.rect(x0, y0, x1 - x0, y1 - y0);
  ctx.fill();
  ctx.stroke();
}

function drawGrid(ctx, transform, terrain) {
  const half = terrainHalf(terrain);
  const major = half <= 8 ? 1 : 2;
  ctx.save();
  ctx.strokeStyle = "rgba(172, 181, 189, 0.13)";
  ctx.lineWidth = 1;
  for (let value = -half; value <= half + 0.001; value += major) {
    const [x0, y0] = worldToScreen(transform, value, -half);
    const [x1, y1] = worldToScreen(transform, value, half);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();

    const [a0, b0] = worldToScreen(transform, -half, value);
    const [a1, b1] = worldToScreen(transform, half, value);
    ctx.beginPath();
    ctx.moveTo(a0, b0);
    ctx.lineTo(a1, b1);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(238, 242, 245, 0.28)";
  ctx.beginPath();
  const [xAxisStartX, xAxisStartY] = worldToScreen(transform, -half, 0);
  const [xAxisEndX, xAxisEndY] = worldToScreen(transform, half, 0);
  const [yAxisStartX, yAxisStartY] = worldToScreen(transform, 0, -half);
  const [yAxisEndX, yAxisEndY] = worldToScreen(transform, 0, half);
  ctx.moveTo(xAxisStartX, xAxisStartY);
  ctx.lineTo(xAxisEndX, xAxisEndY);
  ctx.moveTo(yAxisStartX, yAxisStartY);
  ctx.lineTo(yAxisEndX, yAxisEndY);
  ctx.stroke();
  ctx.restore();
}

function drawArena(ctx, transform, terrain) {
  ctx.save();
  drawSquare(ctx, transform, transform.half, "#11181d", "rgba(221, 226, 232, 0.32)", 1.5);

  if (terrain?.kind === "stepped_arena") {
    for (let step = terrain.step_count; step >= 0; step -= 1) {
      const half = (terrain.center_half_m || 1.5) + step * (terrain.step_width_m || 1.0);
      const color = STEP_COLORS[Math.min(step, STEP_COLORS.length - 1)];
      drawSquare(ctx, transform, half, `${color}22`, `${color}aa`, step === 0 ? 1.3 : 1);
    }
  }

  drawGrid(ctx, transform, terrain);
  ctx.restore();
}

function drawGoal(ctx, transform, goal) {
  if (!Array.isArray(goal) || goal.length < 2) return;
  const [px, py] = worldToScreen(transform, goal[0], goal[1]);
  ctx.save();
  ctx.strokeStyle = "#f0b84f";
  ctx.fillStyle = "rgba(240, 184, 79, 0.16)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(px, py, 14, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(px - 20, py);
  ctx.lineTo(px + 20, py);
  ctx.moveTo(px, py - 20);
  ctx.lineTo(px, py + 20);
  ctx.stroke();
  ctx.restore();
}

function drawTrail(ctx, transform, trail) {
  if (trail.length < 2) return;
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(98, 196, 161, 0.58)";
  ctx.beginPath();
  trail.forEach((point, index) => {
    const [x, y] = worldToScreen(transform, point[0], point[1]);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();
}

function drawBody(ctx, transform, frame, robot) {
  if (!frame?.pos || frame.pos.length < 2) return;
  const x = frame.pos[0];
  const y = frame.pos[1];
  const yaw = frame.rot?.[2] || 0;
  const bodyLength = robot?.body_length_m || 0.28;
  const bodyWidth = robot?.body_width_m || 0.12;
  const halfLength = bodyLength / 2;
  const halfWidth = bodyWidth / 2;
  const corners = [
    [halfLength, halfWidth],
    [halfLength, -halfWidth],
    [-halfLength, -halfWidth],
    [-halfLength, halfWidth],
  ].map(([cx, cy]) => {
    const cos = Math.cos(yaw);
    const sin = Math.sin(yaw);
    return [x + cx * cos - cy * sin, y + cx * sin + cy * cos];
  });

  ctx.save();
  ctx.fillStyle = "#e8edf1";
  ctx.strokeStyle = "#14191d";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  corners.forEach(([wx, wy], index) => {
    const [sx, sy] = worldToScreen(transform, wx, wy);
    if (index === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  });
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  const [noseX, noseY] = worldToScreen(transform, x + halfLength * Math.cos(yaw), y + halfLength * Math.sin(yaw));
  const [centerX, centerY] = worldToScreen(transform, x, y);
  ctx.strokeStyle = "#e06a4f";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(centerX, centerY);
  ctx.lineTo(noseX, noseY);
  ctx.stroke();
  ctx.restore();
}

function drawLegs(ctx, transform, frame, robot) {
  const legs = frame?.legs;
  if (Array.isArray(legs) && legs.length > 0) {
    ctx.save();
    ctx.strokeStyle = "rgba(98, 196, 161, 0.9)";
    ctx.fillStyle = "#62c4a1";
    ctx.lineWidth = 2;
    for (const leg of legs) {
      const mount = leg?.mount;
      const foot = leg?.foot;
      if (!Array.isArray(mount) || !Array.isArray(foot)) continue;
      const [mx, my] = worldToScreen(transform, mount[0], mount[1]);
      const [fx, fy] = worldToScreen(transform, foot[0], foot[1]);
      ctx.beginPath();
      ctx.moveTo(mx, my);
      ctx.lineTo(fx, fy);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(fx, fy, 3.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
    return;
  }

  if (!frame?.pos || !Array.isArray(frame.leg)) return;
  const yaw = frame.rot?.[2] || 0;
  const bodyLength = robot?.body_length_m || 0.28;
  const bodyWidth = robot?.body_width_m || 0.12;
  const legLength = robot?.leg_length_m || 0.16;
  const mounts = [
    [bodyLength / 2, bodyWidth / 2],
    [bodyLength / 2, -bodyWidth / 2],
    [-bodyLength / 2, bodyWidth / 2],
    [-bodyLength / 2, -bodyWidth / 2],
  ];

  ctx.save();
  ctx.strokeStyle = "rgba(98, 196, 161, 0.8)";
  ctx.lineWidth = 2;
  for (let index = 0; index < mounts.length; index += 1) {
    const [mxBody, myBody] = mounts[index];
    const angle = frame.leg[index] || 0;
    const footBody = [mxBody + Math.sin(angle) * legLength, myBody];
    const cos = Math.cos(yaw);
    const sin = Math.sin(yaw);
    const mount = [
      frame.pos[0] + mxBody * cos - myBody * sin,
      frame.pos[1] + mxBody * sin + myBody * cos,
    ];
    const foot = [
      frame.pos[0] + footBody[0] * cos - footBody[1] * sin,
      frame.pos[1] + footBody[0] * sin + footBody[1] * cos,
    ];
    const [mx, my] = worldToScreen(transform, mount[0], mount[1]);
    const [fx, fy] = worldToScreen(transform, foot[0], foot[1]);
    ctx.beginPath();
    ctx.moveTo(mx, my);
    ctx.lineTo(fx, fy);
    ctx.stroke();
  }
  ctx.restore();
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export default function App() {
  const canvasRef = useRef(null);
  const socketRef = useRef(null);
  const frameQueueRef = useRef([]);
  const lastFrameAdvanceRef = useRef(0);
  const streamRef = useRef(null);
  const trailRef = useRef([]);
  const lastTrailFrameRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [metadata, setMetadata] = useState(DEFAULT_METADATA);
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [frame, setFrame] = useState(null);
  const [status, setStatus] = useState(null);
  const [goal, setGoal] = useState(null);
  const [bufferDepth, setBufferDepth] = useState(0);

  function sendMessage(payload) {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(payload));
    }
  }

  function selectModel(modelId) {
    setSelectedModelId(modelId);
    frameQueueRef.current = [];
    trailRef.current = [];
    setFrame(null);
    sendMessage({ type: "select_model", model_id: modelId || null });
  }

  function placeGoal(worldX, worldY) {
    const terrain = metadata.terrain || DEFAULT_METADATA.terrain;
    const half = terrainHalf(terrain);
    const nextGoal = [
      clamp(worldX, -half, half),
      clamp(worldY, -half, half),
      metadata.goal?.height_m || DEFAULT_METADATA.goal.height_m,
    ];
    setGoal(nextGoal);
    frameQueueRef.current = [];
    trailRef.current = [];
    sendMessage({ type: "set_goal", goal: nextGoal });
  }

  function handleCanvasPointerUp(event) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const terrain = metadata.terrain || DEFAULT_METADATA.terrain;
    const transform = mapTransform(rect.width, rect.height, terrain);
    const [worldX, worldY] = screenToWorld(transform, event.clientX - rect.left, event.clientY - rect.top);
    placeGoal(worldX, worldY);
  }

  useEffect(() => {
    let socket;
    let reconnectTimer;

    function connect() {
      socket = new WebSocket(WS_URL);
      socketRef.current = socket;
      socket.onopen = () => setConnected(true);
      socket.onclose = () => {
        setConnected(false);
        reconnectTimer = window.setTimeout(connect, 2000);
      };
      socket.onerror = () => socket.close();
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === "metadata") {
          setMetadata((current) => ({ ...current, ...message }));
        } else if (message.type === "models") {
          setModels(message.models || []);
          setSelectedModelId(message.selected_model_id || "");
        } else if (message.type === "generation") {
          setStatus(message);
          if (Array.isArray(message.goal)) setGoal(message.goal);
          if (message.stream_id && message.stream_id !== streamRef.current) {
            streamRef.current = message.stream_id;
            frameQueueRef.current = [];
            trailRef.current = [];
            setFrame(null);
          }
        } else if (message.type === "goal") {
          if (Array.isArray(message.goal)) setGoal(message.goal);
        } else if (message.type === "frame_batch") {
          if (message.stream_id && message.stream_id !== streamRef.current) {
            streamRef.current = message.stream_id;
            frameQueueRef.current = [];
            trailRef.current = [];
          }
          const frames = Array.isArray(message.frames) ? message.frames : [];
          frameQueueRef.current.push(...frames);
          if (frameQueueRef.current.length > FRAME_BUFFER_LIMIT) {
            frameQueueRef.current.splice(0, frameQueueRef.current.length - FRAME_BUFFER_LIMIT);
          }
          setBufferDepth(frameQueueRef.current.length);
        } else if (message.type === "frame") {
          setFrame(message);
        }
      };
    }

    connect();
    return () => {
      window.clearTimeout(reconnectTimer);
      if (socket) socket.close();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    function tick(timestamp) {
      if (cancelled) return;
      const dtMs = Math.max(16, (metadata.training?.brain_dt_s || DEFAULT_METADATA.training.brain_dt_s) * 1000);
      if (!lastFrameAdvanceRef.current) lastFrameAdvanceRef.current = timestamp;
      if (frameQueueRef.current.length > 0 && timestamp - lastFrameAdvanceRef.current >= dtMs) {
        const nextFrame = frameQueueRef.current.shift();
        lastFrameAdvanceRef.current = timestamp;
        setFrame(nextFrame);
        setBufferDepth(frameQueueRef.current.length);
      }
      window.requestAnimationFrame(tick);
    }

    const requestId = window.requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(requestId);
    };
  }, [metadata.training?.brain_dt_s]);

  useEffect(() => {
    let cancelled = false;

    function render() {
      if (cancelled) return;
      const canvas = canvasRef.current;
      if (!canvas) {
        window.requestAnimationFrame(render);
        return;
      }

      const width = window.innerWidth;
      const height = window.innerHeight;
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;
      const ctx = canvas.getContext("2d");
      const terrain = metadata.terrain || DEFAULT_METADATA.terrain;
      const robot = metadata.robot || DEFAULT_METADATA.robot;
      const transform = mapTransform(width, height, terrain);
      const activeGoal = goal || frame?.goal || status?.goal || metadata.goal?.fixed_goal_xyz;

      if (frame?.pos && frame !== lastTrailFrameRef.current) {
        trailRef.current.push([frame.pos[0], frame.pos[1]]);
        if (trailRef.current.length > TRAIL_LIMIT) trailRef.current.shift();
        lastTrailFrameRef.current = frame;
      }

      ctx.fillStyle = "#0e1114";
      ctx.fillRect(0, 0, width, height);
      drawArena(ctx, transform, terrain);
      drawTrail(ctx, transform, trailRef.current);
      drawGoal(ctx, transform, activeGoal);
      drawLegs(ctx, transform, frame, robot);
      drawBody(ctx, transform, frame, robot);

      if (!frame) {
        ctx.fillStyle = "#9aa4ad";
        ctx.font = "16px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.textAlign = "center";
        ctx.fillText(connected ? "Waiting for replay frames" : `Connecting to ${WS_URL}`, width / 2, height / 2);
        ctx.textAlign = "left";
      }

      window.requestAnimationFrame(render);
    }

    window.requestAnimationFrame(render);
    return () => {
      cancelled = true;
    };
  }, [connected, frame, goal, metadata, status]);

  const activeModel = models.find((model) => model.id === selectedModelId);
  const checkpointLoaded = basename(status?.checkpoint_loaded || activeModel?.checkpoint_path);
  const simulatorBackend = status?.simulator_backend || metadata.simulator?.backend || "unified";
  const viewerResetSeconds = metadata.training?.viewer_reset_s ?? DEFAULT_METADATA.training.viewer_reset_s;
  const remainingReset = Math.max(0, viewerResetSeconds - (frame?.time_s ?? 0));

  return (
    <div className="app-shell">
      <canvas ref={canvasRef} className="arena-canvas" onPointerUp={handleCanvasPointerUp} />

      <aside className="control-panel">
        <div className="panel-header">
          <div>
            <span className="eyebrow">Quadruped Reward Map</span>
            <h1>Live Replay</h1>
          </div>
          <div className="connection-state">
            <span className={connected ? "state-dot state-dot--live" : "state-dot"} />
            {connected ? "live" : "offline"}
          </div>
        </div>

        <label className="field-label" htmlFor="model-select">Model</label>
        <select
          id="model-select"
          className="model-select"
          value={selectedModelId}
          onChange={(event) => selectModel(event.target.value)}
        >
          {models.length === 0 ? (
            <option value="">No saved models</option>
          ) : (
            models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.id}
              </option>
            ))
          )}
        </select>

        <div className="metric-table">
          <div><span>checkpoint</span><strong>{checkpointLoaded}</strong></div>
          <div><span>model type</span><strong>{activeModel?.model_type || metadata.model?.active}</strong></div>
          <div><span>generation</span><strong>{status?.generation ?? activeModel?.generation ?? 0}</strong></div>
          <div><span>best reward</span><strong>{formatNumber(status?.best_reward ?? activeModel?.best_reward)}</strong></div>
          <div><span>mean reward</span><strong>{formatNumber(status?.mean_reward ?? activeModel?.mean_reward)}</strong></div>
          <div><span>buffer</span><strong>{bufferDepth}</strong></div>
          <div><span>reset</span><strong>{formatNumber(remainingReset, 1)}s</strong></div>
          <div><span>backend</span><strong>{simulatorBackend}</strong></div>
        </div>
      </aside>

      <aside className="target-panel">
        <div className="target-row">
          <span>Reward target</span>
          <strong>{goal ? `${formatNumber(goal[0], 2)}, ${formatNumber(goal[1], 2)}` : "none"}</strong>
        </div>
        <div className="target-row">
          <span>Frame time</span>
          <strong>{formatNumber(frame?.time_s, 1)}s</strong>
        </div>
        <div className="target-row">
          <span>Config</span>
          <strong>{metadata.config_name}</strong>
        </div>
      </aside>
    </div>
  );
}
