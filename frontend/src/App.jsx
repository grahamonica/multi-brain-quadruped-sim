import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import "./App.css";

const API_PORT = import.meta.env.VITE_API_PORT || "8000";
const WS_URL =
  import.meta.env.VITE_WS_URL ||
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.hostname}:${API_PORT}/ws`;

const DEFAULT_METADATA = {
  mode: "viewer",
  config_name: "default",
  control: {
    mode: "motor_targets",
  },
  model: {
    active: "shared_trunk_es",
  },
  training: {
    brain_dt_s: 0.05,
  },
  simulator: {
    backend: "mujoco",
  },
  goal: {
    height_m: 0.16,
  },
  terrain: {
    field_half_m: 15.0,
    floor_height_m: 0.0,
  },
  robot: {
    body_length_m: 0.28,
    body_width_m: 0.12,
    body_height_m: 0.02,
    foot_radius_m: 0.01,
  },
};

const FRAME_BUFFER_LIMIT = 400;
const DEFAULT_LEG_COUNT = 4;

function basename(value) {
  if (!value) return "uninitialized";
  const parts = String(value).split("/");
  return parts[parts.length - 1] || value;
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "0.00";
  return Number(value).toFixed(digits);
}

function applyVec3(target, values, fallback) {
  const x = Number(values?.[0]);
  const y = Number(values?.[1]);
  const z = Number(values?.[2]);
  if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
    target.set(x, y, z);
    return;
  }
  target.copy(fallback);
}

export default function App() {
  const viewportRef = useRef(null);
  const sceneRef = useRef(null);
  const socketRef = useRef(null);
  const frameQueueRef = useRef([]);
  const streamRef = useRef(null);
  const awaitingResetRef = useRef(false);
  const lastFrameAdvanceRef = useRef(0);

  const [connected, setConnected] = useState(false);
  const [metadata, setMetadata] = useState(DEFAULT_METADATA);
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [status, setStatus] = useState(null);
  const [frame, setFrame] = useState(null);
  const [bufferDepth, setBufferDepth] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [goalX, setGoalX] = useState("0.0");
  const [goalY, setGoalY] = useState("0.0");

  function sendMessage(payload) {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(payload));
    }
  }

  function selectModel(modelId) {
    setSelectedModelId(modelId);
    awaitingResetRef.current = true;
    frameQueueRef.current = [];
    setBufferDepth(0);
    setFrame(null);
    lastFrameAdvanceRef.current = 0;
    sendMessage({ type: "select_model", model_id: modelId || null });
  }

  function submitGoal(event) {
    event.preventDefault();
    const x = Number(goalX);
    const y = Number(goalY);
    if (Number.isNaN(x) || Number.isNaN(y)) return;
    awaitingResetRef.current = true;
    frameQueueRef.current = [];
    setBufferDepth(0);
    setFrame(null);
    lastFrameAdvanceRef.current = 0;
    sendMessage({
      type: "set_goal",
      goal: [x, y, metadata.goal?.height_m || DEFAULT_METADATA.goal.height_m],
    });
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
          const nextModels = Array.isArray(message.models) ? message.models : [];
          const nextSelectedModelId = String(message.selected_model_id || "");
          const selectedExists = nextSelectedModelId === "" || nextModels.some((model) => model.id === nextSelectedModelId);

          setModels(nextModels);
          setSelectedModelId(selectedExists ? nextSelectedModelId : "");
          if (!selectedExists) {
            sendMessage({ type: "select_model", model_id: null });
          }
        } else if (message.type === "generation") {
          setStatus(message);
          if (message.stream_id && (awaitingResetRef.current || message.stream_id !== streamRef.current)) {
            streamRef.current = message.stream_id;
            frameQueueRef.current = [];
            setBufferDepth(0);
            setFrame(null);
            lastFrameAdvanceRef.current = 0;
          }
          awaitingResetRef.current = false;
        } else if (message.type === "frame_batch") {
          if (awaitingResetRef.current) {
            return;
          }
          if (message.stream_id && message.stream_id !== streamRef.current) {
            return;
          }
          const frames = Array.isArray(message.frames) ? message.frames : [];
          frameQueueRef.current.push(...frames);
          if (frameQueueRef.current.length > FRAME_BUFFER_LIMIT) {
            frameQueueRef.current.splice(0, frameQueueRef.current.length - FRAME_BUFFER_LIMIT);
          }
          setBufferDepth(frameQueueRef.current.length);
        } else if (message.type === "goal") {
          if (Array.isArray(message.goal) && message.goal.length >= 2) {
            setGoalX(String(message.goal[0]));
            setGoalY(String(message.goal[1]));
          }
          awaitingResetRef.current = true;
          frameQueueRef.current = [];
          setBufferDepth(0);
          setFrame(null);
          lastFrameAdvanceRef.current = 0;
        }
      };
    }

    connect();
    const refreshTimer = window.setInterval(() => {
      sendMessage({ type: "refresh_models" });
    }, 2000);
    return () => {
      window.clearTimeout(reconnectTimer);
      window.clearInterval(refreshTimer);
      if (socket) socket.close();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    function tick(timestamp) {
      if (cancelled) return;
      const dtMs = Math.max(
        1,
        ((metadata.training?.brain_dt_s || DEFAULT_METADATA.training.brain_dt_s) * 1000) / playbackSpeed
      );
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
  }, [metadata.training?.brain_dt_s, playbackSpeed]);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070a);

    const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 250);
    camera.position.set(2.2, 1.5, 2.2);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    viewport.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.2);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    scene.add(new THREE.AmbientLight(0xffffff, 0.45));
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
    keyLight.position.set(2.5, 4.0, 1.5);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x8fb5ff, 0.35);
    fillLight.position.set(-3.0, 2.0, -2.0);
    scene.add(fillLight);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshStandardMaterial({ color: 0x0f1a1f, roughness: 0.96, metalness: 0.0 })
    );
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);

    const grid = new THREE.GridHelper(1, 20, 0x2f4b59, 0x1c2d36);
    grid.position.y = 0.001;
    scene.add(grid);

    const body = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({ color: 0x58a6ff, roughness: 0.35, metalness: 0.15 })
    );
    scene.add(body);

    const com = new THREE.Mesh(
      new THREE.SphereGeometry(0.02, 14, 12),
      new THREE.MeshStandardMaterial({ color: 0xf4c95d, emissive: 0x2a1f05, emissiveIntensity: 0.35 })
    );
    scene.add(com);

    const goal = new THREE.Mesh(
      new THREE.SphereGeometry(0.075, 16, 12),
      new THREE.MeshStandardMaterial({ color: 0x66dd88, emissive: 0x113a1e, emissiveIntensity: 0.35 })
    );
    scene.add(goal);

    const legMaterial = new THREE.LineBasicMaterial({ color: 0xa7d6ff });
    const legMountMaterial = new THREE.MeshStandardMaterial({ color: 0x8fc5ff });
    const legFootMaterial = new THREE.MeshStandardMaterial({ color: 0xff9f5a });

    const legObjects = Array.from({ length: DEFAULT_LEG_COUNT }, () => {
      const lineGeometry = new THREE.BufferGeometry();
      lineGeometry.setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, 0], 3));
      const line = new THREE.Line(lineGeometry, legMaterial);

      const mount = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), legMountMaterial);
      mount.scale.setScalar(0.008);

      const foot = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), legFootMaterial);
      foot.scale.setScalar(0.01);

      scene.add(line);
      scene.add(mount);
      scene.add(foot);
      return { line, mount, foot };
    });

    const fallback = new THREE.Vector3(0, 0, 0);

    function resizeRenderer() {
      const width = Math.max(1, viewport.clientWidth);
      const height = Math.max(1, viewport.clientHeight);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    }

    resizeRenderer();
    const resizeObserver = new ResizeObserver(resizeRenderer);
    resizeObserver.observe(viewport);

    let raf = 0;
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      raf = window.requestAnimationFrame(animate);
    };
    animate();

    sceneRef.current = {
      scene,
      camera,
      controls,
      renderer,
      floor,
      grid,
      body,
      com,
      goal,
      legObjects,
      fallback,
    };

    return () => {
      window.cancelAnimationFrame(raf);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      viewport.removeChild(renderer.domElement);
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    const sceneState = sceneRef.current;
    if (!sceneState) return;

    const terrain = metadata.terrain || DEFAULT_METADATA.terrain;
    const robot = metadata.robot || DEFAULT_METADATA.robot;

    const fieldHalf = Math.max(1, Number(terrain.field_half_m || DEFAULT_METADATA.terrain.field_half_m));
    const floorSize = fieldHalf * 2;
    sceneState.floor.scale.set(floorSize, floorSize, 1);
    sceneState.floor.position.y = Number(terrain.floor_height_m || 0);
    sceneState.grid.scale.set(floorSize, floorSize, 1);

    const bodyLength = Math.max(0.05, Number(robot.body_length_m || DEFAULT_METADATA.robot.body_length_m));
    const bodyWidth = Math.max(0.05, Number(robot.body_width_m || DEFAULT_METADATA.robot.body_width_m));
    const bodyHeight = Math.max(0.01, Number(robot.body_height_m || DEFAULT_METADATA.robot.body_height_m));
    sceneState.body.scale.set(bodyLength, bodyWidth, bodyHeight);

    const footRadius = Math.max(0.005, Number(robot.foot_radius_m || DEFAULT_METADATA.robot.foot_radius_m));
    for (const leg of sceneState.legObjects) {
      leg.mount.scale.setScalar(Math.max(0.5 * footRadius, 0.006));
      leg.foot.scale.setScalar(footRadius);
    }
  }, [metadata]);

  useEffect(() => {
    const sceneState = sceneRef.current;
    if (!sceneState || !frame) return;

    const fallback = sceneState.fallback;
    applyVec3(sceneState.body.position, frame.body?.pos, fallback);

    const rotX = Number(frame.body?.rot?.[0]);
    const rotY = Number(frame.body?.rot?.[1]);
    const rotZ = Number(frame.body?.rot?.[2]);
    if (Number.isFinite(rotX) && Number.isFinite(rotY) && Number.isFinite(rotZ)) {
      sceneState.body.rotation.set(rotX, rotY, rotZ, "XYZ");
    }

    applyVec3(sceneState.com.position, frame.com, sceneState.body.position);
    applyVec3(sceneState.goal.position, frame.goal, sceneState.body.position);

    const legs = Array.isArray(frame.legs) ? frame.legs : [];
    for (let index = 0; index < sceneState.legObjects.length; index += 1) {
      const leg = sceneState.legObjects[index];
      const source = legs[index];
      if (!source) {
        leg.line.visible = false;
        leg.mount.visible = false;
        leg.foot.visible = false;
        continue;
      }

      leg.line.visible = true;
      leg.mount.visible = true;
      leg.foot.visible = true;

      const mount = source.mount;
      const foot = source.foot;
      applyVec3(leg.mount.position, mount, sceneState.body.position);
      applyVec3(leg.foot.position, foot, sceneState.body.position);

      const positions = leg.line.geometry.getAttribute("position");
      positions.setXYZ(0, leg.mount.position.x, leg.mount.position.y, leg.mount.position.z);
      positions.setXYZ(1, leg.foot.position.x, leg.foot.position.y, leg.foot.position.z);
      positions.needsUpdate = true;
      leg.line.geometry.computeBoundingSphere();
    }

    const target = sceneState.body.position;
    sceneState.controls.target.lerp(target, 0.08);
  }, [frame]);

  const activeModel = models.find((model) => model.id === selectedModelId);
  const checkpointLoaded = basename(status?.checkpoint_loaded || activeModel?.checkpoint_path);
  const simulatorBackend = status?.simulator_backend || metadata.simulator?.backend || "mujoco";
  const actionMode = metadata.control?.mode || DEFAULT_METADATA.control.mode;
  const loadError =
    Array.isArray(status?.skipped_checkpoints) && status.skipped_checkpoints.length > 0
      ? String(status.skipped_checkpoints[0].reason || "")
      : "";

  return (
    <div className="app">
      <section className="viewport">
        <div ref={viewportRef} className="mujoco-viewport" />
        <div className="viewport-hint">drag orbit, right-drag pan, wheel zoom</div>
      </section>

      <aside className="panel">
        <div className="row title-row">
          <strong>MuJoCo Viewer</strong>
          <span className={connected ? "state state-live" : "state"}>{connected ? "live" : "offline"}</span>
        </div>

        <label htmlFor="model-select" className="label">model</label>
        <select
          id="model-select"
          className="input"
          value={selectedModelId}
          onChange={(event) => selectModel(event.target.value)}
        >
          <option value="">scratch (no checkpoint)</option>
          {models.length === 0 ? (
            <option value="" disabled>No saved models</option>
          ) : (
            models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.id}
              </option>
            ))
          )}
        </select>

        <label htmlFor="playback-speed" className="label">playback speed</label>
        <select
          id="playback-speed"
          className="input"
          value={String(playbackSpeed)}
          onChange={(event) => setPlaybackSpeed(Number(event.target.value))}
        >
          <option value="1">1x</option>
          <option value="2">2x</option>
          <option value="4">4x</option>
          <option value="8">8x</option>
        </select>

        <form className="goal-form" onSubmit={submitGoal}>
          <label className="label" htmlFor="goal-x">goal x,y</label>
          <div className="goal-row">
            <input
              id="goal-x"
              className="input"
              type="number"
              step="0.1"
              value={goalX}
              onChange={(event) => setGoalX(event.target.value)}
            />
            <input
              className="input"
              type="number"
              step="0.1"
              value={goalY}
              onChange={(event) => setGoalY(event.target.value)}
            />
            <button className="button" type="submit">set</button>
          </div>
        </form>

        <div className="stats">
          <div><span>config</span><strong>{metadata.config_name}</strong></div>
          <div><span>backend</span><strong>{simulatorBackend}</strong></div>
          <div><span>checkpoint</span><strong>{checkpointLoaded}</strong></div>
          <div><span>model type</span><strong>{activeModel?.model_type || metadata.model?.active}</strong></div>
          <div><span>generation</span><strong>{status?.generation ?? activeModel?.generation ?? 0}</strong></div>
          <div><span>best reward</span><strong>{formatNumber(status?.best_reward ?? activeModel?.best_reward)}</strong></div>
          <div><span>mean reward</span><strong>{formatNumber(status?.mean_reward ?? activeModel?.mean_reward)}</strong></div>
          <div><span>frame time</span><strong>{formatNumber(frame?.time_s, 2)}s</strong></div>
          <div><span>buffer</span><strong>{bufferDepth}</strong></div>
          <div><span>action mode</span><strong>{actionMode}</strong></div>
          <div><span>command</span><strong>{frame?.selected_command || "-"}</strong></div>
          <div><span>speed</span><strong>{playbackSpeed}x</strong></div>
          <div><span>load error</span><strong>{loadError || "-"}</strong></div>
        </div>
      </aside>
    </div>
  );
}
