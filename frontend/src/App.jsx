import { useEffect, useRef, useState } from "react";
const WS_URL = "ws://localhost:8000/ws";
const FIELD = 15.0;

// ── Orbit camera projection ──────────────────────────────────────────────────
// az = azimuth (rotation around Z, yaw), el = elevation (tilt down from horizontal)
// Default: az=π/4 gives classic 45° isometric look; el=π/4 gives 45° pitch
function project(x, y, z, cam, W, H) {
  const { az, el, scale } = cam;
  const cosAz = Math.cos(az), sinAz = Math.sin(az);
  const cosEl = Math.cos(el), sinEl = Math.sin(el);
  // Rotate world X/Y by azimuth
  const rx = x * cosAz - y * sinAz;
  const ry = x * sinAz + y * cosAz;
  // Project: screen-right = rx, screen-down = ry*cos(el) - z*sin(el)
  const sx = rx * scale;
  const sy = (ry * cosEl - z * sinEl) * scale;
  return [W / 2 + sx, H / 2 + sy];
}

function drawGrid(ctx, cam, W, H) {
  ctx.save();
  ctx.strokeStyle = "#1a2a1a";
  ctx.lineWidth = 0.5;
  const step = 2.0; // 2 m grid lines
  for (let a = -FIELD; a <= FIELD + 0.01; a += step) {
    const [x0, y0] = project(a, -FIELD, 0, cam, W, H);
    const [x1, y1] = project(a,  FIELD, 0, cam, W, H);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    const [x2, y2] = project(-FIELD, a, 0, cam, W, H);
    const [x3, y3] = project( FIELD, a, 0, cam, W, H);
    ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x3, y3); ctx.stroke();
  }
  ctx.strokeStyle = "#2a4a2a";
  ctx.lineWidth = 2;
  const corners = [[-FIELD,-FIELD],[FIELD,-FIELD],[FIELD,FIELD],[-FIELD,FIELD],[-FIELD,-FIELD]];
  ctx.beginPath();
  corners.forEach(([a, b], i) => {
    const [px, py] = project(a, b, 0, cam, W, H);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.restore();
}

function drawGoal(ctx, goal, cam, W, H) {
  const [gx, gy] = project(goal[0], goal[1], 0, cam, W, H);
  const [tx, ty] = project(goal[0], goal[1], goal[2], cam, W, H);
  ctx.save();
  ctx.strokeStyle = "#ffcc00";
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(tx, ty); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#ffcc00";
  ctx.beginPath(); ctx.arc(tx, ty, 8, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "#ffcc0055";
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(gx, gy, 16, 0, Math.PI * 2); ctx.stroke();
  ctx.restore();
}

function drawBody(ctx, body, cam, W, H) {
  if (!body) return;
  const corners = Array.isArray(body.corners) ? body.corners : [];
  if (corners.length !== 8) return;
  const pts = corners.map(point => project(...point, cam, W, H));
  const edges = [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
  ctx.save();
  ctx.strokeStyle = "#8be08b";
  ctx.lineWidth = 2.5;
  edges.forEach(([a, b]) => {
    ctx.beginPath(); ctx.moveTo(pts[a][0], pts[a][1]); ctx.lineTo(pts[b][0], pts[b][1]); ctx.stroke();
  });
  ctx.restore();
}

function drawLegs(ctx, legs, cam, W, H) {
  if (!legs) return;
  legs.forEach(leg => {
    const color = leg.contact_mode === "static" ? "#44ff88"
                : leg.contact_mode === "kinetic" ? "#ff8844"
                : "#446688";
    const [mx, my] = project(...leg.mount, cam, W, H);
    const [fx, fy] = project(...leg.foot, cam, W, H);
    const [lcx, lcy] = project(...leg.com, cam, W, H);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(fx, fy); ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(fx, fy, 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#ffffff55";
    ctx.beginPath(); ctx.arc(lcx, lcy, 3, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  });
}

function drawTrail(ctx, trail, cam, W, H) {
  if (trail.length < 2) return;
  ctx.save();
  ctx.strokeStyle = "#50b4ff";
  ctx.lineWidth = 1.5;
  // Draw in chunks of fading opacity — far fewer stroke() calls
  const CHUNKS = 6;
  const chunkSize = Math.max(1, Math.floor(trail.length / CHUNKS));
  for (let c = 0; c < CHUNKS; c++) {
    const start = c * chunkSize;
    const end = Math.min(trail.length - 1, (c + 1) * chunkSize);
    if (start >= end) continue;
    ctx.globalAlpha = ((c + 1) / CHUNKS) * 0.65;
    ctx.beginPath();
    const [sx, sy] = project(...trail[start], cam, W, H);
    ctx.moveTo(sx, sy);
    for (let i = start + 1; i <= end; i++) {
      const [px, py] = project(...trail[i], cam, W, H);
      ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  ctx.restore();
}

function drawCOM(ctx, com, cam, W, H) {
  if (!com) return;
  const [px, py] = project(...com, cam, W, H);
  ctx.save();
  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#cccccc";
  ctx.font = "11px monospace";
  ctx.fillText("COM", px + 8, py - 6);
  ctx.restore();
}

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
    for (let i = 1; i < data.length; i++) { if (data[i] < min) min = data[i]; if (data[i] > max) max = data[i]; }
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


function drawDistanceChart(ctx, distHistory, W, H) {
  if (distHistory.length < 2) return;
  const cw = Math.min(500, W - 80), ch = 72;
  const ox = (W - cw) / 2, oy = H - ch - 54; // sit just above progress bar

  // Background panel
  ctx.save();
  ctx.fillStyle = "rgba(10,24,10,0.88)";
  ctx.strokeStyle = "#223322";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(ox - 12, oy - 22, cw + 24, ch + 30, 6);
  ctx.fill();
  ctx.stroke();

  // Label
  ctx.fillStyle = "#556655";
  ctx.font = "10px monospace";
  ctx.textAlign = "left";
  ctx.fillText("distance to goal (m)", ox, oy - 10);

  // Current value + start value
  const cur = distHistory[distHistory.length - 1];
  const start = distHistory[0];
  const delta = start - cur;
  ctx.fillStyle = delta >= 0 ? "#44ff88" : "#ff6644";
  ctx.textAlign = "right";
  ctx.fillText(`${cur.toFixed(2)} m  (${delta >= 0 ? "−" : "+"}${Math.abs(delta).toFixed(2)})`, ox + cw, oy - 10);
  ctx.textAlign = "left";

  // Chart line
  let maxD = 0.1;
  for (let i = 0; i < distHistory.length; i++) if (distHistory[i] > maxD) maxD = distHistory[i];
  ctx.strokeStyle = "#ffaa33";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  distHistory.forEach((v, i) => {
    const px = ox + (i / (distHistory.length - 1)) * cw;
    const py = oy + ch - (v / maxD) * ch;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();

  // Filled area under line
  ctx.lineTo(ox + cw, oy + ch);
  ctx.lineTo(ox, oy + ch);
  ctx.closePath();
  ctx.fillStyle = "rgba(255,170,51,0.08)";
  ctx.fill();

  // Y-axis max label
  ctx.fillStyle = "#445544";
  ctx.font = "9px monospace";
  ctx.fillText(maxD.toFixed(1), ox - 10, oy + 4);
  ctx.fillText("0", ox - 10, oy + ch);

  ctx.restore();
}

function drawMotors(ctx, motors, W, H) {
  if (!motors || motors.length === 0) return;
  const panelW = 220;
  const panelH = 22 + motors.length * 22;
  const x = 18;
  const y = H - panelH - 54;
  const maxVel = 8.0;

  ctx.save();
  ctx.fillStyle = "rgba(10,24,10,0.88)";
  ctx.strokeStyle = "#223322";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, panelW, panelH, 6);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#88cc88";
  ctx.font = "10px monospace";
  ctx.fillText("motor target velocity (rad/s)", x + 10, y + 14);

  motors.forEach((motor, index) => {
    const rowY = y + 22 + index * 22;
    const barX = x + 96;
    const barW = 108;
    const zeroX = barX + barW / 2;
    const vel = Math.max(-maxVel, Math.min(maxVel, motor.target_velocity_rad_s ?? 0));
    const fillW = Math.abs(vel / maxVel) * (barW / 2);

    ctx.fillStyle = "#556655";
    ctx.fillText(motor.name, x + 10, rowY + 10);
    ctx.strokeStyle = "#334433";
    ctx.beginPath();
    ctx.moveTo(zeroX, rowY - 2);
    ctx.lineTo(zeroX, rowY + 10);
    ctx.stroke();
    ctx.strokeStyle = "#223322";
    ctx.strokeRect(barX, rowY, barW, 8);
    ctx.fillStyle = vel >= 0 ? "#44ff88" : "#ff8844";
    ctx.fillRect(vel >= 0 ? zeroX : zeroX - fillW, rowY + 1, fillW, 6);
  });

  ctx.restore();
}

export default function App() {
  const canvasRef = useRef(null);
  const stepQueueRef = useRef([]);    // buffered step msgs — consumed at real-time rate
  const currentStepRef = useRef(null); // last rendered step (held when queue empty)
  const animRef = useRef(null);
  const camRef = useRef(null);
  const trailRef = useRef([]);      // array of [x, y, z] world positions
  const distRef = useRef([]);       // distance-to-goal per step this episode
  const lastStepRef = useRef(-1);   // detect episode resets
  const [genInfo, setGenInfo] = useState(null);
  const [connected, setConnected] = useState(false);

  // Initialise camera lazily on first render
  function getCam(W, H) {
    if (!camRef.current) {
      camRef.current = {
        az: Math.PI / 4,
        el: Math.PI / 4,
        scale: Math.min(W, H) / (FIELD * Math.sqrt(2) * 1.4),
        dragging: false,
        lastX: 0,
        lastY: 0,
      };
    }
    return camRef.current;
  }

  // Mouse / wheel handlers on the canvas element
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    function onMouseDown(e) {
      const cam = camRef.current;
      if (!cam) return;
      cam.dragging = true;
      cam.lastX = e.clientX;
      cam.lastY = e.clientY;
    }
    function onMouseMove(e) {
      const cam = camRef.current;
      if (!cam || !cam.dragging) return;
      const dx = e.clientX - cam.lastX;
      const dy = e.clientY - cam.lastY;
      cam.az -= dx * 0.005;
      cam.el  = Math.max(0.05, Math.min(Math.PI * 0.48, cam.el + dy * 0.005));
      cam.lastX = e.clientX;
      cam.lastY = e.clientY;
    }
    function onMouseUp() {
      if (camRef.current) camRef.current.dragging = false;
    }
    function onWheel(e) {
      const cam = camRef.current;
      if (!cam) return;
      e.preventDefault();
      cam.scale *= e.deltaY > 0 ? 0.93 : 1.07;
      cam.scale = Math.max(2, Math.min(600, cam.scale));
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

  useEffect(() => {
    let ws;
    function connect() {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => setConnected(true);
      ws.onclose = () => { setConnected(false); setTimeout(connect, 2000); };
      ws.onerror = () => ws.close();
      ws.onmessage = e => {
        const msg = JSON.parse(e.data);
        if (msg.type === "step") {
          // Reset trail + distance history at episode boundaries
          if (msg.step < lastStepRef.current) {
            trailRef.current = [];
            distRef.current = [];
          }
          lastStepRef.current = msg.step;
          if (msg.com) {
            trailRef.current.push(msg.com);
            if (trailRef.current.length > 600) trailRef.current.shift();
          }
          if (msg.com && msg.goal) {
            const dx = msg.com[0] - msg.goal[0], dy = msg.com[1] - msg.goal[1];
            distRef.current.push(Math.sqrt(dx * dx + dy * dy));
          }
          // Buffer for real-time playback — hold up to 2 full episodes
          stepQueueRef.current.push(msg);
          if (stepQueueRef.current.length > 6000) stepQueueRef.current.shift();
        } else if (msg.type === "generation") {
          setGenInfo(msg);
        }
      };
    }
    connect();
    return () => ws && ws.close();
  }, []);

  useEffect(() => {
    let isCancelled = false;

    function render() {
      if (isCancelled) return;

      const canvas = canvasRef.current;
      if (!canvas) {
        animRef.current = requestAnimationFrame(render);
        return;
      }
      const W = window.innerWidth, H = window.innerHeight;
      if (canvas.width !== W) canvas.width = W;
      if (canvas.height !== H) canvas.height = H;
      const ctx = canvas.getContext("2d");
      const cam = getCam(W, H);

      ctx.fillStyle = "#060e06";
      ctx.fillRect(0, 0, W, H);
      drawGrid(ctx, cam, W, H);

      // Always drain the full queue and show the latest step — training always
      // runs faster than real-time, so there's no benefit to real-time throttling.
      const queue = stepQueueRef.current;
      if (queue.length > 0) {
        currentStepRef.current = queue[queue.length - 1];
        queue.length = 0;
      }
      const step = currentStepRef.current;

      if (step) {
        drawGoal(ctx, step.goal, cam, W, H);
        drawTrail(ctx, trailRef.current, cam, W, H);
        drawBody(ctx, step.body, cam, W, H);
        drawLegs(ctx, step.legs, cam, W, H);
        drawCOM(ctx, step.com, cam, W, H);

        drawMotors(ctx, step.motors, W, H);
        drawDistanceChart(ctx, distRef.current, W, H);

        const pct = step.step / step.total_steps;
        const queued = stepQueueRef.current.length;
        ctx.fillStyle = "#0d1f0d";
        ctx.fillRect(20, H - 32, W - 40, 10);
        ctx.fillStyle = "#33cc66";
        ctx.fillRect(20, H - 32, (W - 40) * pct, 10);
        ctx.fillStyle = "#aaa";
        ctx.font = "11px monospace";
        ctx.fillText(
          `episode ${(pct * 100).toFixed(0)}%   reward ${step.reward.toFixed(3)}   sim ${step.time_s.toFixed(2)}s` +
          (step.noise_scale != null ? `   noise ${step.noise_scale.toFixed(2)}` : "") +
          (step.closing_rate_m_s != null ? `   close ${step.closing_rate_m_s.toFixed(2)}m/s` : "") +
          (queued > 10 ? `   [+${queued}]` : ""),
          24, H - 38
        );
      } else {
        drawGoal(ctx, [1, 0, 0.16], cam, W, H);
        ctx.fillStyle = "#335533";
        ctx.font = "18px monospace";
        ctx.textAlign = "center";
        ctx.fillText(connected ? "Waiting for first episode…" : "Connecting to ws://localhost:8000…", W / 2, H / 2);
        ctx.textAlign = "left";
      }

      animRef.current = requestAnimationFrame(render);
    }

    animRef.current = requestAnimationFrame(render);
    return () => {
      isCancelled = true;
      cancelAnimationFrame(animRef.current);
    };
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
          SNN Quadruped Training
          <span style={{ marginLeft: 10, fontSize: 11, color: connected ? "#44ff44" : "#ff4444" }}>
            ● {connected ? "live" : "offline"}
          </span>
        </div>
        {genInfo ? (
          <>
            <div>generation      <b style={{ color: "#aaffaa" }}>{genInfo.generation}</b></div>
            {(genInfo.top_rewards || []).map((r, i) => (
              <div key={i}>top {i + 1} reward    <b style={{ color: "#aaffaa" }}>{r.toFixed(3)}</b></div>
            ))}
            <div>mean reward     <b style={{ color: "#aaffaa" }}>{genInfo.mean_reward.toFixed(3)}</b></div>
            <div>goal xyz        <b style={{ color: "#ffcc88" }}>[{genInfo.goal.map(v => v.toFixed(2)).join(", ")}]</b></div>
            <div style={{ marginTop: 10, marginBottom: 4, color: "#556655", fontSize: 10 }}>mean reward / generation</div>
            <RewardChart history={genInfo.rewards_history} />
          </>
        ) : (
          <div style={{ color: "#446644" }}>—</div>
        )}
        <div style={{ marginTop: 12, color: "#334433", fontSize: 10, lineHeight: 1.6 }}>
          inputs 48 = goal·3 + COM·3 + body·3 + feet·12 + leg-COM·12 + body-IMU·3 + leg-IMU·12<br />
          hidden 4 layers = shared trunk 64 + 4 motor lanes of 64 per layer (τ=20ms)<br />
          ES: pop=8  σ=0.05  lr=0.03  episode=30s  field=30m<br />
          drag to orbit · scroll to zoom
        </div>
      </div>


      {/* Legend */}
      <div style={{
        position: "absolute", bottom: 50, right: 16,
        background: "#0a180a99", border: "1px solid #1a2a1a",
        borderRadius: 6, padding: "8px 14px", color: "#88cc88", fontSize: 11,
        pointerEvents: "none",
      }}>
        <span style={{ color: "#44ff88" }}>●</span> static&ensp;
        <span style={{ color: "#ff8844" }}>●</span> kinetic&ensp;
        <span style={{ color: "#446688" }}>●</span> airborne&ensp;
        <span style={{ color: "#ffcc00" }}>★</span> goal
      </div>
    </div>
  );
}
