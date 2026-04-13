import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
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
    field_half_m: 15.0,
    center_half_m: 2.5,
    step_count: 5,
    step_width_m: 2.0,
    step_height_m: 0.15,
    floor_height_m: 0.0,
  },
  robot: {
    body_length_m: 0.28,
    body_width_m: 0.12,
    body_height_m: 0.02,
    leg_length_m: 0.16,
    leg_radius_m: 0.01,
    foot_radius_m: 0.01,
  },
  model: {
    active: "shared_trunk_es",
    architecture: "shared_trunk_motor_lanes",
    trainer: "openai_es",
    registered: [],
  },
  goal: {
    strategy: "radial_random",
    radius_m: 10.0,
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
    backend: "mujoco",
  },
};

const STEP_COLORS = [0x225c48, 0x2f6c56, 0x3b7d64, 0x4a8d73, 0x5ea786, 0x70bf98];
const TRAIL_LIMIT = 220;
const FRAME_BUFFER_LIMIT = 900;
const CLICK_MOVE_TOLERANCE_PX = 6;

function basename(value) {
  if (!value) return "uninitialized";
  const parts = String(value).split("/");
  return parts[parts.length - 1] || value;
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "0.00";
  return Number(value).toFixed(digits);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function terrainHalf(terrain) {
  return Math.max(terrain?.field_half_m || DEFAULT_METADATA.terrain.field_half_m, 1.0);
}

function terrainHeightAt(terrain, x, y) {
  const activeTerrain = terrain || DEFAULT_METADATA.terrain;
  const floor = Number(activeTerrain.floor_height_m || 0.0);
  if (activeTerrain.kind === "flat") return floor;
  const radius = Math.max(Math.abs(x), Math.abs(y));
  const centerHalf = Number(activeTerrain.center_half_m || 0);
  if (radius <= centerHalf) return floor;
  const stepWidth = Math.max(Number(activeTerrain.step_width_m || 1.0), 1e-6);
  const rawStep = (radius - centerHalf) / stepWidth;
  const stepIndex = Math.min(Math.max(Math.floor(rawStep + 1e-6), 0), Number(activeTerrain.step_count || 0));
  return floor + stepIndex * Number(activeTerrain.step_height_m || 0.0);
}

function worldToSceneVector(x = 0, y = 0, z = 0) {
  return new THREE.Vector3(Number(x), Number(z), -Number(y));
}

function disposeObject3D(root) {
  root.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (Array.isArray(child.material)) {
      for (const material of child.material) material.dispose();
    } else if (child.material) {
      child.material.dispose();
    }
  });
}

function buildTerrainGroup(terrain) {
  const activeTerrain = terrain || DEFAULT_METADATA.terrain;
  const group = new THREE.Group();
  const floorHeight = Number(activeTerrain.floor_height_m || 0.0);
  const half = terrainHalf(activeTerrain);

  const base = new THREE.Mesh(
    new THREE.PlaneGeometry(half * 2, half * 2),
    new THREE.MeshStandardMaterial({ color: 0x111920, roughness: 0.95, metalness: 0.05 })
  );
  base.rotation.x = -Math.PI / 2;
  base.position.y = floorHeight;
  base.receiveShadow = true;
  group.add(base);

  const boundaryPoints = [
    worldToSceneVector(-half, -half, floorHeight + 0.002),
    worldToSceneVector(half, -half, floorHeight + 0.002),
    worldToSceneVector(half, half, floorHeight + 0.002),
    worldToSceneVector(-half, half, floorHeight + 0.002),
    worldToSceneVector(-half, -half, floorHeight + 0.002),
  ];
  const boundary = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(boundaryPoints),
    new THREE.LineBasicMaterial({ color: 0xdde2e8 })
  );
  group.add(boundary);

  if (activeTerrain.kind === "stepped_arena") {
    for (let level = 1; level <= Number(activeTerrain.step_count || 0); level += 1) {
      const inner = Number(activeTerrain.center_half_m || 0) + ((level - 1) * Number(activeTerrain.step_width_m || 1));
      const outer = Number(activeTerrain.center_half_m || 0) + (level * Number(activeTerrain.step_width_m || 1));
      const top = floorHeight + (level * Number(activeTerrain.step_height_m || 0));
      const halfHeight = Math.max((top - floorHeight) * 0.5, 0.001);
      const centerZ = floorHeight + halfHeight;
      const stripHalf = Math.max((outer - inner) * 0.5, 0.001);
      const color = STEP_COLORS[Math.min(level - 1, STEP_COLORS.length - 1)];
      const material = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.9,
        metalness: 0.03,
        transparent: true,
        opacity: 0.88,
      });

      const north = new THREE.Mesh(new THREE.BoxGeometry(outer * 2, halfHeight * 2, stripHalf * 2), material);
      north.position.copy(worldToSceneVector(0, (inner + outer) * 0.5, centerZ));
      north.receiveShadow = true;
      north.castShadow = true;
      group.add(north);

      const south = new THREE.Mesh(new THREE.BoxGeometry(outer * 2, halfHeight * 2, stripHalf * 2), material.clone());
      south.position.copy(worldToSceneVector(0, -((inner + outer) * 0.5), centerZ));
      south.receiveShadow = true;
      south.castShadow = true;
      group.add(south);

      const east = new THREE.Mesh(new THREE.BoxGeometry(stripHalf * 2, halfHeight * 2, inner * 2), material.clone());
      east.position.copy(worldToSceneVector((inner + outer) * 0.5, 0, centerZ));
      east.receiveShadow = true;
      east.castShadow = true;
      group.add(east);

      const west = new THREE.Mesh(new THREE.BoxGeometry(stripHalf * 2, halfHeight * 2, inner * 2), material.clone());
      west.position.copy(worldToSceneVector(-((inner + outer) * 0.5), 0, centerZ));
      west.receiveShadow = true;
      west.castShadow = true;
      group.add(west);
    }
  }

  const gridDivisions = Math.max(Math.round(half * 2), 8);
  const grid = new THREE.GridHelper(half * 2, gridDivisions, 0x7f95a8, 0x394754);
  grid.position.y = floorHeight + 0.004;
  group.add(grid);

  return group;
}

function buildRobotGroup(robot) {
  const activeRobot = robot || DEFAULT_METADATA.robot;
  const group = new THREE.Group();

  const body = new THREE.Mesh(
    new THREE.BoxGeometry(
      Number(activeRobot.body_length_m || DEFAULT_METADATA.robot.body_length_m),
      Number(activeRobot.body_height_m || DEFAULT_METADATA.robot.body_height_m),
      Number(activeRobot.body_width_m || DEFAULT_METADATA.robot.body_width_m)
    ),
    new THREE.MeshStandardMaterial({ color: 0xced7df, metalness: 0.08, roughness: 0.35 })
  );
  body.castShadow = true;
  body.receiveShadow = true;
  group.add(body);

  const nose = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0.2, 0, 0)]),
    new THREE.LineBasicMaterial({ color: 0xe5714f })
  );
  group.add(nose);

  const legVisuals = [];
  for (let i = 0; i < 4; i += 1) {
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineBasicMaterial({ color: 0x62c4a1 })
    );
    const foot = new THREE.Mesh(
      new THREE.SphereGeometry(Math.max(Number(activeRobot.foot_radius_m || 0.01), 0.008), 12, 12),
      new THREE.MeshStandardMaterial({ color: 0x62c4a1, roughness: 0.45, metalness: 0.05 })
    );
    const mount = new THREE.Mesh(
      new THREE.SphereGeometry(0.008, 10, 10),
      new THREE.MeshStandardMaterial({ color: 0x88d5be, roughness: 0.55, metalness: 0.02 })
    );
    foot.castShadow = true;
    mount.castShadow = true;
    group.add(line);
    group.add(foot);
    group.add(mount);
    legVisuals.push({ line, foot, mount });
  }

  group.userData.body = body;
  group.userData.nose = nose;
  group.userData.legs = legVisuals;
  return group;
}

function applyRobotFrame(robotGroup, frame, axisTransform) {
  if (!robotGroup || !frame) return;

  const body = robotGroup.userData.body;
  const nose = robotGroup.userData.nose;
  const legs = robotGroup.userData.legs || [];

  const pos = Array.isArray(frame.pos) ? frame.pos : [0, 0, 0];
  const rot = Array.isArray(frame.rot) ? frame.rot : [0, 0, 0];
  body.position.copy(worldToSceneVector(pos[0], pos[1], pos[2]));

  const worldQuat = new THREE.Quaternion().setFromEuler(
    new THREE.Euler(Number(rot[0] || 0), Number(rot[1] || 0), Number(rot[2] || 0), "XYZ")
  );
  const sceneQuat = axisTransform.toScene.clone().multiply(worldQuat).multiply(axisTransform.toWorld);
  body.quaternion.copy(sceneQuat);

  const halfLength = body.geometry.parameters.width * 0.5;
  nose.geometry.setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(halfLength, 0, 0)]);
  nose.position.copy(body.position);
  nose.quaternion.copy(body.quaternion);

  const frameLegs = Array.isArray(frame.legs) ? frame.legs : [];
  for (let i = 0; i < legs.length; i += 1) {
    const visual = legs[i];
    const leg = frameLegs[i];
    if (!leg || !Array.isArray(leg.mount) || !Array.isArray(leg.foot)) {
      visual.line.visible = false;
      visual.foot.visible = false;
      visual.mount.visible = false;
      continue;
    }

    const mount = worldToSceneVector(leg.mount[0], leg.mount[1], leg.mount[2]);
    const foot = worldToSceneVector(leg.foot[0], leg.foot[1], leg.foot[2]);
    visual.line.visible = true;
    visual.foot.visible = true;
    visual.mount.visible = true;
    visual.line.geometry.setFromPoints([mount, foot]);
    visual.foot.position.copy(foot);
    visual.mount.position.copy(mount);
  }
}

function applyGoal(goalMesh, goal) {
  if (!goalMesh) return;
  if (!Array.isArray(goal) || goal.length < 2) {
    goalMesh.visible = false;
    return;
  }

  const gx = Number(goal[0]);
  const gy = Number(goal[1]);
  const gz = Number(goal[2] || 0.0);
  goalMesh.visible = true;
  goalMesh.position.copy(worldToSceneVector(gx, gy, gz + 0.01));
}

function applyTrail(trailLine, trail, terrain) {
  if (!trailLine) return;
  if (!Array.isArray(trail) || trail.length < 2) {
    trailLine.visible = false;
    return;
  }

  const points = trail.map(([x, y]) => {
    const z = terrainHeightAt(terrain, x, y) + 0.03;
    return worldToSceneVector(x, y, z);
  });
  trailLine.visible = true;
  trailLine.geometry.setFromPoints(points);
}

export default function App() {
  const viewportRef = useRef(null);
  const socketRef = useRef(null);
  const frameQueueRef = useRef([]);
  const lastFrameAdvanceRef = useRef(0);
  const streamRef = useRef(null);
  const trailRef = useRef([]);
  const lastTrailFrameRef = useRef(null);
  const sceneRef = useRef(null);
  const pointerGestureRef = useRef({ x: 0, y: 0, moved: false });

  const [connected, setConnected] = useState(false);
  const [metadata, setMetadata] = useState(DEFAULT_METADATA);
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [frame, setFrame] = useState(null);
  const [status, setStatus] = useState(null);
  const [goal, setGoal] = useState(null);
  const [bufferDepth, setBufferDepth] = useState(0);

  const axisTransform = useMemo(() => {
    const toScene = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
    return { toScene, toWorld: toScene.clone().invert() };
  }, []);

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

  function pickWorldGoal(clientX, clientY) {
    const state = sceneRef.current;
    if (!state) return;

    const rect = state.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );

    state.raycaster.setFromCamera(ndc, state.camera);
    const floor = Number((metadata.terrain || DEFAULT_METADATA.terrain).floor_height_m || 0);
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -floor);
    const hit = new THREE.Vector3();
    if (!state.raycaster.ray.intersectPlane(plane, hit)) return;

    const worldX = hit.x;
    const worldY = -hit.z;
    placeGoal(worldX, worldY);
  }

  function handlePointerDown(event) {
    pointerGestureRef.current = { x: event.clientX, y: event.clientY, moved: false };
  }

  function handlePointerMove(event) {
    const dx = event.clientX - pointerGestureRef.current.x;
    const dy = event.clientY - pointerGestureRef.current.y;
    if ((dx * dx) + (dy * dy) > (CLICK_MOVE_TOLERANCE_PX * CLICK_MOVE_TOLERANCE_PX)) {
      pointerGestureRef.current.moved = true;
    }
  }

  function handlePointerUp(event) {
    if (!pointerGestureRef.current.moved) {
      pickWorldGoal(event.clientX, event.clientY);
    }
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
    const container = viewportRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1319);

    const camera = new THREE.PerspectiveCamera(52, 1, 0.05, 250);
    camera.position.set(9, 6.4, 9.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0.6, 0);
    controls.minDistance = 1.4;
    controls.maxDistance = 70;
    controls.maxPolarAngle = Math.PI * 0.495;

    const ambient = new THREE.HemisphereLight(0xb8d4f1, 0x263241, 0.82);
    scene.add(ambient);

    const key = new THREE.DirectionalLight(0xffffff, 0.95);
    key.position.set(12, 16, 8);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.5;
    key.shadow.camera.far = 60;
    key.shadow.camera.left = -22;
    key.shadow.camera.right = 22;
    key.shadow.camera.top = 22;
    key.shadow.camera.bottom = -22;
    scene.add(key);

    const world = new THREE.Group();
    scene.add(world);

    const terrainGroup = buildTerrainGroup(metadata.terrain || DEFAULT_METADATA.terrain);
    world.add(terrainGroup);

    const trailLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineBasicMaterial({ color: 0x62c4a1, transparent: true, opacity: 0.72 })
    );
    trailLine.visible = false;
    world.add(trailLine);

    const robotGroup = buildRobotGroup(metadata.robot || DEFAULT_METADATA.robot);
    world.add(robotGroup);

    const goalGroup = new THREE.Group();
    const goalRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.24, 0.03, 12, 42),
      new THREE.MeshStandardMaterial({ color: 0xf0b84f, emissive: 0xa36f1d, emissiveIntensity: 0.25 })
    );
    goalRing.rotation.x = Math.PI / 2;
    const goalPin = new THREE.Mesh(
      new THREE.CylinderGeometry(0.02, 0.02, 0.32, 10),
      new THREE.MeshStandardMaterial({ color: 0xf0b84f, roughness: 0.3, metalness: 0.08 })
    );
    goalPin.position.y = 0.16;
    goalGroup.add(goalRing);
    goalGroup.add(goalPin);
    goalGroup.visible = false;
    world.add(goalGroup);

    const raycaster = new THREE.Raycaster();

    function resize() {
      const rect = container.getBoundingClientRect();
      const width = Math.max(1, Math.floor(rect.width));
      const height = Math.max(1, Math.floor(rect.height));
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    }

    resize();
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);

    let rafId = 0;
    const renderLoop = () => {
      controls.update();
      renderer.render(scene, camera);
      rafId = window.requestAnimationFrame(renderLoop);
    };
    rafId = window.requestAnimationFrame(renderLoop);

    sceneRef.current = {
      scene,
      camera,
      renderer,
      controls,
      world,
      terrainGroup,
      trailLine,
      robotGroup,
      goalGroup,
      raycaster,
      resizeObserver,
      rafId,
    };

    return () => {
      window.cancelAnimationFrame(rafId);
      resizeObserver.disconnect();
      controls.dispose();
      disposeObject3D(scene);
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
      if (sceneRef.current?.scene === scene) {
        sceneRef.current = null;
      }
    };
  }, [axisTransform, metadata.robot, metadata.terrain]);

  useEffect(() => {
    const state = sceneRef.current;
    if (!state) return;

    const activeGoal = goal || frame?.goal || status?.goal || metadata.goal?.fixed_goal_xyz;

    if (frame?.pos && frame !== lastTrailFrameRef.current) {
      trailRef.current.push([frame.pos[0], frame.pos[1]]);
      if (trailRef.current.length > TRAIL_LIMIT) trailRef.current.shift();
      lastTrailFrameRef.current = frame;
    }

    applyRobotFrame(state.robotGroup, frame, axisTransform);
    applyTrail(state.trailLine, trailRef.current, metadata.terrain || DEFAULT_METADATA.terrain);
    applyGoal(state.goalGroup, activeGoal);

    if (frame?.pos) {
      const anchor = worldToSceneVector(frame.pos[0], frame.pos[1], frame.pos[2] || 0);
      state.controls.target.lerp(anchor, 0.18);
    }
  }, [axisTransform, frame, goal, metadata.goal?.fixed_goal_xyz, metadata.terrain, status?.goal]);

  const activeModel = models.find((model) => model.id === selectedModelId);
  const checkpointLoaded = basename(status?.checkpoint_loaded || activeModel?.checkpoint_path);
  const simulatorBackend = status?.simulator_backend || metadata.simulator?.backend || "mujoco";
  const viewerResetSeconds = metadata.training?.viewer_reset_s ?? DEFAULT_METADATA.training.viewer_reset_s;
  const remainingReset = Math.max(0, viewerResetSeconds - (frame?.time_s ?? 0));

  return (
    <div className="app-shell">
      <div
        ref={viewportRef}
        className="viewer-viewport"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
      />

      <aside className="control-panel">
        <div className="panel-header">
          <div>
            <span className="eyebrow">Quadruped 3D Viewer</span>
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
        <div className="target-row">
          <span>Controls</span>
          <strong>drag orbit, click set goal</strong>
        </div>
      </aside>
    </div>
  );
}
