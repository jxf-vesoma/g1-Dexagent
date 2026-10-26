# setup_g1_dex1: Unitree G1 arms with Dex1-1 grippers

A DexAgent setup beside the default bench in `setup/`. It has the Unitree G1's two 7-DOF arms, each with a Unitree
Dex1-1 parallel-jaw gripper on its wrist, at a table. The robot's own head camera is the ego view. The layout is the one
`setup/` uses: the rig frame, the `robot_<side>_` naming, the floating-wrist proposal `scene.xml` and the rigid runtime
that `tool_library.common.rigid_robot.compile_rigid` compiles from it. So the pipeline's stages run on it unchanged.
`setup/` itself is untouched.

| file | what |
|---|---|
| `scene.xml` | proposal scene: table, G1 torso (visual only), both arms, both grippers on virtual wrists, keyframe `home`, cameras `ego`, `wrist_left`, `wrist_right` |
| `setup.yaml` | the setup, machine-readable (the keys of `setup/setup.yaml`, plus the jaw joints and the measured work area) |
| `robot/dex1_<side>.xml` | one gripper alone on the six-joint virtual wrist: the `hand_xml` of the grasp tools |
| `robot/g1_arm_<side>.xml` | one arm alone, shoulder to wrist flange: the inertias the runtime restores |
| `assets/` | the meshes, copied unchanged (`assets/LICENSES.md`: provenance and sha256), and `test_cup.xml` |
| `images/` | home-pose views (`setup/render_setup.py`) and the cup grasp (`check_setup.py`) |
| `build_setup.py` | the generator. Every file above comes from it; edit it, not them |
| `check_setup.py` | counts, jaw and arm tracking on the physical runtime, top-down cup lift with each gripper |

## Regenerate and check

```bash
python setup_g1_dex1/build_setup.py --models <vesoma-models checkout>
MUJOCO_GL=egl python setup_g1_dex1/check_setup.py
```

`build_setup.py` reads `robots/g1_with_dex1/g1_with_dex1.mujoco.xml` from the checkout. Its meshes must be real files:
run `git lfs pull` there first, or the generator stops at the first Git LFS pointer. The output is deterministic: two
runs write byte-identical files.

Use the repository's pinned environment (`requirements.txt`): `mujoco==3.10.0` and `numpy>=1.26,<2`. The setup was
generated and checked with mujoco 3.10.0, numpy 1.26.4 and scipy 1.17.1.

## How the robot is modelled

- **Arms.** Kinematics, inertias, joint limits, armature and friction come from the source model. Arm motor torques are
  capped at the G1's actuator limits: 25 N·m for the shoulder, elbow and wrist roll, 5 N·m for wrist pitch and yaw.
  `scene.xml` declares the arm to the runtime compiler through custom fields `dexagent_arm_*_<side>`: joints,
  torque caps, nullspace posture gains, armature, damping and the inertia file. `rigid_robot.trusted_arms` reads them
  from the setup scene, never from a proposal. A setup without them is the YAM bench.
- **Grippers.** The hand frame's +z points along the jaws (the approach), and the jaws close along its y. The two jaws
  are slides, `<side>_thumb_jaw` (the source's finger 1) and `<side>_index_jaw` (finger 2). Their axes are reversed from
  the source so that, as for every DexAgent hand, closing moves toward the high end of the range: 0.022938 is closed,
  −0.021562 is open (88.9 mm between the pads). One equality `<side>_jaw_coupling` makes the index jaw follow the thumb
  jaw, as the single motor does on the real gripper. Each jaw has the source's position servo: kp 2000 N/m, kv 100 N·s/m,
  capped at 20 N. `hand.py` recognises the gripper from those two joint names (`hand_spec_for`, `jaw_roles`).
- **Collision.** Each gripper has five collision geoms: the casting, the wrist camera housing and barrel, and one per
  pad. The pads use the source's simplified pad meshes. The source's arm collision geoms are dropped; the runtime builds
  hulls of the arm links itself, as for the YAM.
- **Layout.** The torso faces +y with the shoulder pitch axes 0.30 m above the tabletop; the table's near edge is
  0.10 m in front of them. At home each gripper rests low beside the work, like the demonstrator's hand on the
  table: 23 cm out to its side, 6 cm forward and 8 cm up, pointing into the work 10° down, with each arm kept at least
  5 cm off the torso.

## Known limitations

- **Short reach.** With the gripper pointing straight down and its pad centre 5 cm above the table, an arm reaches
  only y = 0.10 to 0.30 m. Both arms reach only near the centreline, x from −0.10 to 0.10 m (`setup.yaml`
  `table.work_area`). Parts further out have to be grasped from the side or with the gripper tilted.
- **Ego view.** The ego camera is the G1 head camera: 42.5° vertical field of view, pitched 47.6° down. At home the
  grippers appear only in the bottom corners of the image (`images/ego_home.png`).
- **Jaw deadband.** The jaws settle about 0.1 mm short of a commanded position, which is the joint friction (0.2 N)
  over the servo stiffness. A fully closed command reaches 0.02284 of 0.02294.
- **Wrist roll tracking.** The planned sandbox issue of up to 0.05 rad wrist-roll error does not occur here. The
  worst arm joint error while the grippers close at home is 5e-5 rad (`check_setup.py`). Larger errors under load have
  not been measured.
- **Visual-only torso.** The torso and head (`robot_torso`) are visual only, and the model has no pelvis or legs. The
  runtime pairs arm with arm, arm with gripper, gripper with gripper, and each of them with the table, but nothing
  stops an arm from passing through the chest. Plans that bring an elbow behind the shoulders need their own check.
