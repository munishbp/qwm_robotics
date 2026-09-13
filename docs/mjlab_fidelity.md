# Physics fidelity options for the mjlab task

`docs/mjlab_port.md` section 5 lists the simplifications of the port. `docs/next_steps.md` section
"Physics fidelity and scale" ranks the work that removes them. `MjlabConfig` now carries four
options. Every one defaults to the behavior of the port, so an existing result stays reproducible.
`MjlabConfig.__post_init__` rejects an unknown value.

| Field | Default | Other value | Item in `next_steps.md` |
|---|---|---|---|
| `robot_collision` | `False` | `True` | 3 |
| `lift` | `"central"` | `"at_latch"` | 1, stage B |
| `latch` | `"kinematic"` | `"connect"` | 5 |
| `pusher_shape` | `"cylinder"` | `"box"` | 4 |

## 1. `robot_collision`

A robot geom uses `contype=2, conaffinity=1`, so no robot pair matches. With the option on both
masks become 3, so a robot pair matches on bit 0 and bit 1. The payload (`contype=5,
conaffinity=5`) still matches on bit 0. The floor and the walls use bit 2 alone, so a robot still
passes through them and the joint ranges still hold it in the arena. Two pushers 2 m apart drive
into each other, 5.5 m from the payload. The smallest centre to centre gap over 20 control steps
is **0.042 m** with the option off and **0.371 m** with it on. Two discs of radius 0.2 m touch at
0.4 m, so the option stops them at contact with 2.9 cm of solver softness.

## 2. `lift`

A latched gripper unloads the friction with an upward force. `"central"` writes the total into
`xfrc_applied[:, payload, 2]`, which acts at the centre of mass and cannot tip the box.
`"at_latch"` splits the same total between the latched grippers and adds the moment `r x F` about
the centre of mass, where `r` runs from the centre to the latch point. The 80 percent of weight
bound stays on the total, so four latched grippers still cannot lift the box off the floor.

One gripper latched at the middle of the +x face, 20 control steps, `lift_force` 25 N, resting
height 0.15 m. `central` gives height 0.14993 m, roll 0.0000 deg, pitch 0.0000 deg. `at_latch`
gives height 0.14993 m, roll 0.0000 deg, pitch **0.0024 deg**. The margin explains it: 25 N at
`hx` 0.8 m gives 20 N.m, and the remaining 53.5 N of weight restores 42.8 N.m.

The same test at `lift_force` 45 N separates the options. `central` holds the pitch at 0.000 deg.
`at_latch` tips the box to 8.557 deg and raises it to 0.267 m. So the moment reaches the payload,
and the default 25 N sits below the tipping threshold.

## 3. `latch`

`"kinematic"` writes the gripper `qpos` every substep and pulls the payload through
`xfrc_applied`. `"connect"` gives every gripper an `mjEQ_CONNECT` equality to the payload,
inactive at compile. A latch writes `eq_active` and `eq_data` per world. The gripper is no longer
teleported and applies no external force. It drives with its own velocity servo, limited to 3 N,
and the constraint carries that force. The unloading force stays as it is.

What I verified in the source. `Data.eq_active` has shape `(nworld, neq)`
(`mujoco_warp/_src/types.py` line 2229), so a toggle is an in place write through the
`WarpBridge`. `Model.eq_data` has shape `("*", "neq", vec11)` (`types.py` line 1744) and every
equality kernel reads `eq_data[worldid % eq_data.shape[0], eqid]` (`constraint.py` lines 239, 562,
707, 1058, 1521), so after expansion each world reads its own row.
`Simulation.expand_model_fields(("eq_data",))` tiles the field and calls `create_graph()` once
(`mjlab/sim/sim.py` lines 434 to 448), so the env pays the recapture at construction only. The
connect kernel reads `anchor1 = data[0:3]` in body 1 and `anchor2 = data[3:6]` in body 2
(`constraint.py` lines 239 to 256) and costs three rows. `mjwarp.reset_data` restores `eq_active`
from `eq_active0` (`io.py` line 2504), but the env never calls `Simulation.reset`, so
`_write_state` writes `eq_active` itself on every reset. A latch would otherwise leak into the
next episode.

Before writing the code I ran the per world write on the GPU: four worlds, two latched at opposite
anchors and two free, 100 physics steps. The two latched worlds held their anchor to 0.13 mm and
the free worlds ran away. So the anchor is settable per world here, and the fixed anchor fallback
is not needed.

**The anchor.** `eq_obj1id` is the gripper body and `eq_obj2id` is the payload. The gripper side
anchor stays at the body origin, so a latch writes the payload side anchor only. That anchor is
the latch point plus one robot radius along the outward normal, in payload coordinates: the point
the kinematic latch used to write. It rotates with the payload, which a gripper side anchor could
not, because a gripper body has two slide joints and no rotational freedom. For the same reason
the port uses `mjEQ_CONNECT` and not `mjEQ_WELD`: a weld would pin the payload yaw to the world
through one gripper, and it costs six rows instead of three.

**Measurements.** Two pushers on the -x face, two grippers latched on the +x face, all commanded
toward +x. The largest anchor error is **1.19 mm** over 100 physics steps and **1.67 mm** over
200, while the payload travels 0.201 m and 0.478 m. The friction facts hold: two latched grippers
alone move it **0.0000 m**, and two pushers with two latched grippers move it **0.478 m**, above
the 0.3 m gate.

## 4. `pusher_shape`

`"box"` gives a pusher a box of half extents `(robot_radius, robot_radius, robot_height / 2)`.
`docs/next_steps.md` item 4 states that the multicontact expansion in
`mujoco_warp/_src/collision_convex.py` admits BOX and MESH only, so a cylinder against a box gets
one point, and MuJoCo Warp prints that warning at construction. The largest contact count between
the pusher geom and the payload geom over 20 control steps confirms it. At payload yaw 0.00 rad a
cylinder gets **1** point and a box gets **4**. At yaw 0.25 rad a cylinder gets **1** and a box
gets **2**.

**The rotation result is the opposite of the prediction.** Two pushers press the same half of the
long face of a 2 kg payload for 20 control steps. The cylinder pushers turn it **1.003 rad**. The
box pushers turn it **0.054 rad**, a factor of 19 less. A sweep over five geometries (short face,
long face, opposed couple, and a payload pre rotated to 0.25 rad) gives the same direction every
time. The mechanism is the contact patch: one point cannot resist a turn, and a patch of two to
four points carries a couple. So a box pusher holds the payload yaw instead of letting it spin. It
gains the yaw authority item 4 asks for, but it does not give more yaw.
`test_box_pusher_resists_yaw_that_a_cylinder_pusher_cannot` records the measured direction.

## Approximations that remain

- The 2D geometry code still models every robot as a disc of radius 0.2 m. A box pusher reaches
  0.283 m at a corner, so `rect_contact`, the latch test, and the occlusion test are 8 cm
  optimistic there.
- With `latch="connect"` the gripper adds 2 kg of inertia and drags with up to 3 N when the
  payload outruns its command. The same team moves the payload 1.153 m with the kinematic latch
  and 0.478 m with the connect latch over 20 control steps, so a policy does not transfer between
  the two without a retrain. A gripper at a joint range limit also fights its own constraint,
  which happens against a wall only.
- `lift="at_latch"` still applies a lift that falls by decree. It moves where the lift acts.
- `_solver_capacity` raises `nconmax` and `njmax` by the robot pair count, four points per pair
  for a box pusher and one for a cylinder, plus three rows per gripper equality. The default team
  of six reaches 48 and 174 with every option on. MuJoCo Warp drops a contact above the budget
  without an error, so check the numbers before a larger team.
- I measured the control step cost at 8 envs only, where launch latency dominates and the spread
  is 0.96x to 1.08x. That is not a throughput number. Measure at 256 envs before a run.

## Running the pipeline with an option

The options reach the task through the `mj` argument of `MjlabTransportEnv`. Two call sites build
the task: `scripts/common.py` line 59 (`make_env`, used by the training and evaluation scripts)
and `scripts/run_controls.py` line 102. Neither passes `mj`, so both use the defaults today. To
expose the options, each call site passes `mj=MjlabConfig(...)`. The smallest change reads them in
`scripts/common.py` next to `SWARM_SIM`, for example from `SWARM_MJ_COLLIDE`, `SWARM_MJ_LIFT`,
`SWARM_MJ_LATCH` and `SWARM_MJ_PUSHER`. `scripts/run_controls.py` then needs the same lines or an
import of that helper. I did not edit either script.

```
.venv/bin/python -m pytest tests/test_env_mjlab.py -q
```
