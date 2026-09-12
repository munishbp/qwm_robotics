# The task on mjlab (MuJoCo Warp)

`swarm/env_mjlab.py` runs the transport task on mjlab with the interface of `design.md`
section 3.6, so every script, the buffer, the belief pipeline, and the search run unchanged with
`SWARM_SIM=mjlab`. This document records the mapping from the 2D task, the constants, and the
measurements of the port.

## 1. Mapping

| 2D task (`swarm/env.py`) | mjlab task (`swarm/env_mjlab.py`) |
|---|---|
| Arena, half size 5 m, robots clamped | Plane floor and four static wall boxes; robot slide joints limited to ±4.8 m |
| Payload rectangle 0.8 by 0.4 m, quasi static | Free box 0.8 by 0.4 by 0.15 m, 8 kg, Coulomb friction 0.5 with the floor, elliptic cone, impedance ratio 10 |
| Friction threshold `3.5 - 1.0 n_latched`, floor 0.5 (2D units) | Static friction `0.5 (78.5 N - 25 N n_latched)`: 39 N with no latch, 27 N with one, 14 N with two |
| Pusher force 1.0 along the inward normal when `u > 0` in contact | Cylinder on two slide joints with velocity servos limited to 10 N per axis; with `u > 0` in contact the command adds an inward component, so the robot presses on the face and the contact force is real |
| Gripper latches at the boundary point, pulls with 0.3 in any direction | Latches kinematically: the body is written to one radius outside the boundary point every substep, the pull of at most 3 N acts on the payload at the boundary point as an external force with its torque, and 25 N upward per latch at the payload center unloads the friction, bounded at 80 percent of the weight so the box never leaves the floor |
| Scout applies no force | Servo limited to 3 N; it cannot move the payload |
| Robots do not collide with each other | Collision groups: robots collide with the payload only, the payload with the floor and walls |
| Time step 0.1 s | Physics step 0.01 s, decimation 10 |
| Robot speed 1.5 m/s, instantaneous | Servo target speed 2.5 m/s, gain 60, 2 kg body |

The friction facts of the design hold in the new physics (`tests/test_env_mjlab.py`): one
pusher and three pushers cannot move the payload, two latched grippers alone cannot, and two
pushers with two latched grippers move it 0.3 m or more in two seconds.

Everything above the physics is inherited: the observation layout, the sensing ranges, the
occlusion test, the message mask, the reset sampling, the success test, and the decoder target.

## 2. Why these choices

- **Kinematic latching instead of a weld constraint.** A MuJoCo weld holds the relative pose
  stored in the model, which is a model field, not a per world data field. Latching at an
  arbitrary pose per env would need per world model expansion and a graph recapture on every
  latch. Writing the gripper's position every substep gives the same behavior the 2D task had.
- **Unloading instead of lowering a threshold.** The 2D task lowered the friction threshold per
  latch by fiat. In real physics a latched gripper that lifts one end of the payload lowers the
  normal force. The lift is applied at the payload center so the box does not tip.
- **Pushing through contact.** The pusher's force is the actuator force transmitted through the
  cylinder to box contact, limited by the actuator's force range. This is the part of the physics
  the 2D task could only imitate.
- **A higher top speed.** A velocity servo needs time to accelerate and stops short when its
  force limit binds against the payload. At 1.5 m/s the scripted controller reaches 52 percent;
  at 2.5 m/s it reaches 75 percent at 64 envs.

## 3. Measurements

| Quantity | Value |
|---|---|
| First construction (Warp kernel compilation, cached afterwards) | 60 s |
| Construction with a warm cache | about 5 s |
| Throughput at 64, 128, 256 envs | 2,900, 6,100, 11,850 env steps per second |
| GPU memory of the process at 4 to 256 envs | 718 MiB, flat |
| Scripted controller, 256 envs, speed 2.5 m/s | 83.6 percent success, 78.9 steps on success; single pusher and single gripper 0 |
| Throughput at 256 envs in the controls script | 14,400 env steps per second |
| Belief training, 24,000 steps at 256 envs | 66 minutes, best evaluation 48.4 percent |

For comparison the 2D simulator runs 41,600 env steps per second. Training at 256 envs spends
most of its time in the learner (about 0.15 s per step with 4 updates), so the slower physics
adds about 15 percent to a run.

## 4. Running it

```
uv pip install --python .venv/bin/python -e ".[mjlab]"
.venv/bin/python -m pytest tests/test_env_mjlab.py -q
SWARM_SIM=mjlab RUN_DIR=$PWD/runs/mjlab bash scripts/run_all.sh 24000
```

Memory safety: `swarm/compute.py` polls the driver for this process's GPU memory every three
seconds and exits the process above 8 GB, because MuJoCo Warp allocates outside the PyTorch
allocator that the fraction cap controls.

## 5. Known differences that can change results

- The unloading force was unbounded in the first version. Four latched grippers lifted the box
  off the floor and the transfer to large teams degraded. The bound fixed it and the transfer
  cells were rerun (`results.md` section 13.7).

- Contact is soft. Three pushers at 30 N against a 34 N threshold crept at 5 cm/s in the first
  configuration, which is why the margin was widened to 39 N.
- A cylinder against a box gets at most one contact point in MuJoCo Warp's collision detection,
  so a pusher cannot apply a torque through its own contact patch.
- Rotation of the payload comes from off center pushes and pulls against the four contact
  points of the box on the floor, not from a torque threshold.
