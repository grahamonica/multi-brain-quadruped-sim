import { useEffect, useRef, useState } from "react";
const WS_URL = "ws://localhost:8000/ws";

// Arena constants (must match jax_trainer.py)
const ARENA_CENTER_HALF = 2.5;
const STEP_WIDTH = 2.0;
const STEP_HEIGHT = 0.15;
const N_STEPS = 5;

// Bot geometry (must match jax_trainer.py)
const BODY_L = 0.14;   // half body length
const BODY_W = 0.06;   // half body width
const LEG_LEN = 0.16;
// Mount points in body frame [x, y, z]: front-left, front-right, rear-left, rear-right
const MOUNTS_BODY = [
  [ BODY_L,  BODY_W, 0],
  [ BODY_L, -BODY_W, 0],
  [-BODY_L,  BODY_W, 0],
  [-BODY_L, -BODY_W, 0],
];

// Step-level colours: 0=center → 5=escaped
const STEP_COLORS = ["#334433", "#44aa55", "#88cc44", "#ffcc00", "#ff8844", "#ff44aa"];
const BOT_COLORS  = ["#33ff88", "#66ff44", "#ffee00", "#ff8833", "#ff3388", "#ff33ff"];

// ── Rotation matrix from euler xyz (roll, pitch, yaw) ───────────────────────
function rotMat(rot) {
  const [r, p, y] = rot;
  const cr = Math.cos(r), sr = Math.sin(r);
  const cp = Math.cos(p), sp = Math.sin(p);
  const cy = Math.cos(y), sy = Math.sin(y);
  return [
    [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
    [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
    [-sp,      cp * sr,                  cp * cr],
  ];
}

function applyRot(R, v) {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
  ];
}

// ── Orbit camera projection ───────────────────────────────────────────────────
function project(x, y, z, cam, W, H) {
  const { az, el, scale } = cam;
  const cosAz = Math.cos(az), sinAz = Math.sin(az);
  const cosEl = Math.cos(el), sinEl = Math.sin(el);
  const rx = x * cosAz - y * sinAz;
  const ry = x * sinAz + y * cosAz;
  const sx = rx * scale;
  const sy = (ry * cosEl - z * sinEl) * scale;
  return [W / 2 + sx, H / 2 + sy];
}

// ── Arena (stepped terrain) ───────────────────────────────────────────────────
function drawArena(ctx, cam, W, H) {
  ctx.save();
  for (let s = N_STEPS; s >= 0; s--) {
    const r = ARENA_CENTER_HALF + s * STEP_WIDTH;
    const h = s * STEP_HEIGHT;
    const corners = [[-r, -r], [r, -r], [r, r], [-r, r]];
    const pts = corners.map(([cx, cy]) => project(cx, cy, h, cam, W, H));

    ctx.fillStyle = s === 0 ? "#0c1a0c" : STEP_COLORS[Math.min(s, STEP_COLORS.length - 1)] + "22";
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = s === 0 ? "#1a3a1a" : STEP_COLORS[Math.min(s, STEP_COLORS.length - 1)] + "55";
    ctx.lineWidth = s === 0 ? 1 : 1.5;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.stroke();

    if (s > 0) {
      const rPrev = ARENA_CENTER_HALF + (s - 1) * STEP_WIDTH;
      const hPrev = (s - 1) * STEP_HEIGHT;
      const innerCorners = [[-rPrev, -rPrev], [rPrev, -rPrev], [rPrev, rPrev], [-rPrev, rPrev]];
      const inner = innerCorners.map(([cx, cy]) => project(cx, cy, h, cam, W, H));
      const innerLow = innerCorners.map(([cx, cy]) => project(cx, cy, hPrev, cam, W, H));
      ctx.strokeStyle = STEP_COLORS[Math.min(s, STEP_COLORS.length - 1)] + "66";
      ctx.lineWidth = 1;
      for (let i = 0; i < 4; i++) {
        ctx.beginPath();
        ctx.moveTo(inner[i][0], inner[i][1]);
        ctx.lineTo(innerLow[i][0], innerLow[i][1]);
        ctx.stroke();
      }
    }
  }
  ctx.restore();
}

// ── Bot wireframe ─────────────────────────────────────────────────────────────
// Body corners in body frame (top face only, z=0 at mount height)
const BODY_CORNERS_BODY = [
  [-BODY_L, -BODY_W, 0],
  [ BODY_L, -BODY_W, 0],
  [ BODY_L,  BODY_W, 0],
  [-BODY_L,  BODY_W, 0],
];

function drawSwarm(ctx, swarm, cam, W, H) {
  if (!swarm || !swarm.n) return;
  const { pos, rot, leg, level, n } = swarm;

  // Group bots by step level for batched stroke calls
  const bodyPaths = Array.from({ length: BOT_COLORS.length }, () => []);
  const legPaths  = Array.from({ length: BOT_COLORS.length }, () => []);

  for (let i = 0; i < n; i++) {
    const bi = i * 3;
    const li = i * 4;
    const bx = pos[bi], by = pos[bi + 1], bz = pos[bi + 2];
    const rx = rot[bi], ry = rot[bi + 1], rz = rot[bi + 2];
    const lvl = Math.min(Math.round(level[i] || 0), BOT_COLORS.length - 1);

    const R = rotMat([rx, ry, rz]);

    // Body corners in world space
    const wcorners = BODY_CORNERS_BODY.map(v => {
      const w = applyRot(R, v);
      return project(bx + w[0], by + w[1], bz + w[2], cam, W, H);
    });
    bodyPaths[lvl].push(wcorners);

    // Leg segments: mount → foot
    const legSeg = [];
    for (let j = 0; j < 4; j++) {
      const mb = MOUNTS_BODY[j];
      const mw = applyRot(R, mb);
      const mx = bx + mw[0], my = by + mw[1], mz = bz + mw[2];

      const angle = leg[li + j];
      const fb = [LEG_LEN * Math.sin(angle), 0, -LEG_LEN * Math.cos(angle)];
      const fw = applyRot(R, fb);
      const fx = mx + fw[0], fy = my + fw[1], fz = mz + fw[2];

      legSeg.push([
        project(mx, my, mz, cam, W, H),
        project(fx, fy, fz, cam, W, H),
      ]);
    }
    legPaths[lvl].push(legSeg);
  }

  // Draw legs first (behind bodies)
  for (let lvl = 0; lvl < BOT_COLORS.length; lvl++) {
    if (!legPaths[lvl].length) continue;
    ctx.strokeStyle = BOT_COLORS[lvl] + "88";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    for (const segs of legPaths[lvl]) {
      for (const [[x1, y1], [x2, y2]] of segs) {
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }
    }
    ctx.stroke();
  }

  // Draw bodies
  for (let lvl = 0; lvl < BOT_COLORS.length; lvl++) {
    if (!bodyPaths[lvl].length) continue;
    ctx.strokeStyle = BOT_COLORS[lvl];
    ctx.lineWidth = lvl > 0 ? 1.2 : 0.8;
    ctx.beginPath();
    for (const corners of bodyPaths[lvl]) {
      ctx.moveTo(corners[0][0], corners[0][1]);
      for (let k = 1; k < corners.length; k++) ctx.lineTo(corners[k][0], corners[k][1]);
      ctx.closePath();
    }
    ctx.stroke();
  }
}

// ── Reward chart ──────────────────────────────────────────────────────────────
function RewardChart({ history }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    const data = history.slice(-100);
    if (!c || data.length < 2) return;
    const ctx = c.getContext("2d");
    const w = c.width, h = c.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0a120a";
    ctx.fillRect(0, 0, w, h);
    let min = data[0], max = data[0];
    for (let i = 1; i < data.length; i++) {
      if (data[i] < min) min = data[i];
      if (data[i] > max) max = data[i];
    }
    const range = max - min || 1;
    ctx.strokeStyle = "#33ff66";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    data.forEach((v, i) => {
      const px = (i / (data.length - 1)) * w;
      const py = h - ((v - min) / range) * (h - 8) - 4;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
    if (min < 0 && max > 0) {
      const zy = h - ((0 - min) / range) * (h - 8) - 4;
      ctx.strokeStyle = "#334433";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(0, zy); ctx.lineTo(w, zy); ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [history]);
  return <canvas ref={ref} width={320} height={90} style={{ width: "100%", height: 90 }} />;
}

// ── Status HUD on canvas ───────────────────────────────────────────────────────
function drawHUD(ctx, swarm, W, H) {
  if (!swarm) return;
  ctx.fillStyle = "#aaa";
  ctx.font = "11px monospace";
  ctx.fillText(
    `gen ${swarm.gen || 0}   bots ${swarm.n || 0}`,
    24, H - 20,
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const canvasRef   = useRef(null);
  const swarmRef    = useRef(null);
  const animRef     = useRef(null);
  const camRef      = useRef(null);
  const [genInfo, setGenInfo]     = useState(null);
  const [connected, setConnected] = useState(false);

  function getCam(W, H) {
    if (!camRef.current) {
      camRef.current = {
        az: Math.PI / 4,
        el: Math.PI / 5,
        scale: Math.min(W, H) / 24,
        dragging: false, lastX: 0, lastY: 0,
      };
    }
    return camRef.current;
  }

  // Mouse / wheel handlers
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    function onMouseDown(e) {
      const cam = camRef.current;
      if (!cam) return;
      cam.dragging = true; cam.lastX = e.clientX; cam.lastY = e.clientY;
    }
    function onMouseMove(e) {
      const cam = camRef.current;
      if (!cam || !cam.dragging) return;
      cam.az -= (e.clientX - cam.lastX) * 0.005;
      cam.el  = Math.max(0.05, Math.min(Math.PI * 0.48, cam.el + (e.clientY - cam.lastY) * 0.005));
      cam.lastX = e.clientX; cam.lastY = e.clientY;
    }
    function onMouseUp() { if (camRef.current) camRef.current.dragging = false; }
    function onWheel(e) {
      const cam = camRef.current;
      if (!cam) return;
      e.preventDefault();
      cam.scale *= e.deltaY > 0 ? 0.93 : 1.07;
      cam.scale = Math.max(2, Math.min(800, cam.scale));
    }
    canvas.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      canvas.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      canvas.removeEventListener("wheel", onWheel);
    };
  }, []);

  // WebSocket
  useEffect(() => {
    let ws;
    function connect() {
      ws = new WebSocket(WS_URL);
      ws.onopen  = () => setConnected(true);
      ws.onclose = () => { setConnected(false); setTimeout(connect, 2000); };
      ws.onerror = () => ws.close();
      ws.onmessage = e => {
        const msg = JSON.parse(e.data);
        if (msg.type === "swarm") {
          swarmRef.current = msg;
        } else if (msg.type === "generation") {
          setGenInfo(msg);
        }
      };
    }
    connect();
    return () => ws && ws.close();
  }, []);

  // Render loop
  useEffect(() => {
    let cancelled = false;
    function render() {
      if (cancelled) return;
      const canvas = canvasRef.current;
      if (!canvas) { animRef.current = requestAnimationFrame(render); return; }
      const W = window.innerWidth, H = window.innerHeight;
      if (canvas.width !== W) canvas.width = W;
      if (canvas.height !== H) canvas.height = H;
      const ctx = canvas.getContext("2d");
      const cam = getCam(W, H);

      ctx.fillStyle = "#060e06";
      ctx.fillRect(0, 0, W, H);

      drawArena(ctx, cam, W, H);

      const swarm = swarmRef.current;
      drawSwarm(ctx, swarm, cam, W, H);
      drawHUD(ctx, swarm, W, H);

      if (!swarm) {
        ctx.fillStyle = "#335533";
        ctx.font = "18px monospace";
        ctx.textAlign = "center";
        ctx.fillText(
          connected ? "Waiting for first swarm generation…" : "Connecting to ws://localhost:8000…",
          W / 2, H / 2,
        );
        ctx.textAlign = "left";
      }

      animRef.current = requestAnimationFrame(render);
    }
    animRef.current = requestAnimationFrame(render);
    return () => { cancelled = true; cancelAnimationFrame(animRef.current); };
  }, [connected]);

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#060e06", overflow: "hidden", fontFamily: "monospace" }}>
      <canvas ref={canvasRef} style={{ position: "absolute", inset: 0, cursor: "grab" }} />

      {/* HUD */}
      <div style={{
        position: "absolute", top: 16, right: 16,
        background: "#0a180add", border: "1px solid #223322",
        borderRadius: 8, padding: "14px 18px", width: 340, color: "#88cc88",
        fontSize: 12, lineHeight: 1.8, pointerEvents: "none",
      }}>
        <div style={{ fontSize: 15, fontWeight: "bold", color: "#44ff88", marginBottom: 8 }}>
          Swarm Arena Training
          <span style={{ marginLeft: 10, fontSize: 11, color: connected ? "#44ff44" : "#ff4444" }}>
            ● {connected ? "live" : "offline"}
          </span>
        </div>
        {genInfo ? (
          <>
            <div>generation      <b style={{ color: "#aaffaa" }}>{genInfo.generation}</b></div>
            {(genInfo.top_rewards || []).map((r, i) => (
              <div key={i}>top {i + 1} reward    <b style={{ color: "#aaffaa" }}>{r.toFixed(1)}</b></div>
            ))}
            <div>mean reward     <b style={{ color: "#aaffaa" }}>{genInfo.mean_reward.toFixed(1)}</b></div>
            <div style={{ marginTop: 10, marginBottom: 4, color: "#556655", fontSize: 10 }}>mean reward / generation</div>
            <RewardChart history={genInfo.rewards_history} />
          </>
        ) : (
          <div style={{ color: "#446644" }}>—</div>
        )}
        <div style={{ marginTop: 12, color: "#334433", fontSize: 10, lineHeight: 1.6 }}>
          arena 5×5m center · 5 steps × 2m · 0.15m/step<br />
          continuous respawn · per-bot outward goals · 30s lifespan<br />
          drag to orbit · scroll to zoom
        </div>
      </div>

      {/* Step-level legend */}
      <div style={{
        position: "absolute", bottom: 50, right: 16,
        background: "#0a180a99", border: "1px solid #1a2a1a",
        borderRadius: 6, padding: "8px 14px", color: "#88cc88", fontSize: 11,
        pointerEvents: "none",
      }}>
        {BOT_COLORS.map((c, i) => (
          <span key={i} style={{ marginRight: 8 }}>
            <span style={{ color: c }}>●</span> {i === 0 ? "center" : i === N_STEPS ? "escaped" : `step ${i}`}
          </span>
        ))}
      </div>
    </div>
  );
}
