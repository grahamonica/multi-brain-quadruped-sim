# AI Quadruped v2

Minimal Python quadruped simulation inspired by the KT2 Kungfu Turtle layout.

## Physics Scope

- The floor is an immovable plane at `z = 0`.
- Gravity is always applied using `F = ma`.
- The body is simulated as a 3D rigid body with translation and roll/pitch/yaw rotation.
- Each foot can enter `airborne`, `static`, or `kinetic` contact in 3D.
- Contact forces at the feet and body corners generate both linear force and tipping torque.
- If the center of mass moves outside the active support region, the robot can rotate and fall.

## Run

```bash
python3 main.py
```

This opens a small Tkinter demo with four motor sliders, a live canvas view, and force telemetry.
