# Stage 4: Robot Data Generation

Stage 3's verified robot trajectory and the human video in; randomized, photorealistic robot episodes out, each with its joint trajectory, ready for policy training.

| Step | What it does | Runs on |
|---|---|---|
| 1. `randomize` | Draws new initial object and robot states, adapts the trajectory to each one, and keeps only the episodes that pass the feasibility checks and the physics replay. | CPU |
| 2. `plate` | Removes the demonstrator's hands and the manipulated objects from the human video with SAM 3 and [VOID](https://github.com/netflix/void-model), giving a clean background plate. | GPU |
| 3. `export` | Writes a Blender scene for every episode and camera view: geometry, materials, per-frame poses, cameras. | CPU |
| 4. `render` | Renders the robot and the objects with Blender Cycles, from the calibrated ego camera and from any extra views such as wrist cameras. | GPU |
| 5. `composite` | Pastes the renders onto the clean plate and encodes one video per view. | CPU |
| 6. `lerobot` (optional) | Writes the episodes as a [LeRobot](https://github.com/huggingface/lerobot) v2.1 dataset, e.g. for training GR00T or other VLA policies. See [LeRobot export](#lerobot-export). | CPU |

## Requirements

- **OS**: Linux, with an NVIDIA GPU and a recent driver.
- **GPU memory**: **40 GB or more** for the `plate` step. VOID loads a 5B-parameter video model and peaked at 38 GB in our runs on an RTX A6000 (48 GB). The `render` step needs about 8 GB.
- **Disk**: about 35 GB for model weights and 9 GB for the Python environment.
- **Python**: 3.11 exactly. The Blender wheel only exists for 3.11.
- **Blender**: you do **not** need to install Blender. It comes as the `bpy` Python package and is installed by `setup.sh`.

## Installation

```bash
conda create -y -n dexagent-stage4 python=3.11
conda activate dexagent-stage4
cd stage4_robot_data_generation
bash setup.sh
```

`setup.sh` does three things:

1. clones VOID into `third_party/void-model`,
2. installs the Python packages (VOID's pinned requirements, SAM 3, MuJoCo, Blender),
3. downloads the model weights listed below into the Hugging Face cache.

### Models to download in advance

All three are needed by the `plate` step. `setup.sh` downloads them; the table is here so you can fetch them yourself on a machine without internet access, or check that they are in place.

| Model | Hugging Face repo | File | Size | Access |
|---|---|---|---|---|
| VOID pass 1 | [`netflix/void-model`](https://huggingface.co/netflix/void-model) | `void_pass1.safetensors` | 11 GB | open |
| CogVideoX-Fun (VOID's base model) | [`alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP) | whole repo | 20 GB | open |
| SAM 3.1 | [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1) | `sam3.1_multiplex.pt`, `config.json` | 3.5 GB | **gated** |

SAM 3.1 is gated. Request access on its model page, then log in before running `setup.sh`:

```bash
hf auth login
```

To download by hand:

```bash
hf download netflix/void-model void_pass1.safetensors
hf download alibaba-pai/CogVideoX-Fun-V1.5-5b-InP
hf download facebook/sam3.1 sam3.1_multiplex.pt config.json
```

The weights go to the Hugging Face cache (`$HF_HOME`, default `~/.cache/huggingface`). Set `HF_HOME` if you want them somewhere else. If VOID is cloned elsewhere, point `VOID_DIR` at it.

### Blackwell GPUs (RTX 50 series, RTX PRO 6000 Blackwell, B200)

VOID's requirements install PyTorch 2.7.1 built for CUDA 12.6, which has no kernels for Blackwell GPUs (compute capability 10.0 or 12.0). The `plate` step then fails with

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

After `setup.sh`, replace PyTorch and torchvision with the CUDA 12.8 builds of the same versions:

```bash
pip install --force-reinstall --no-deps torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

The first command swaps the two packages; the second installs the CUDA 12.8 libraries they need. Check with

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(), torch.cuda.get_arch_list())"
```

which should print `2.7.1+cu128` and list your GPU's architecture (e.g. `sm_120`). It needs an NVIDIA driver that supports CUDA 12.8 (570 or newer). Re-run the two commands whenever `setup.sh` reinstalls VOID's requirements.

### `pyarrow requires NumPy 2.0 or newer`

VOID pins `numpy==1.26.4`, but `datasets` can pull in a `pyarrow` built for NumPy 2. VOID then fails while importing `transformers` (`plate` step, `outputs/plate/void.log` ends with `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4`). `requirements.txt` pins `pyarrow<20`; in an environment installed before that pin, run

```bash
pip install "pyarrow==19.0.1"
```

## Inputs

Stage 3 writes this folder for you (`runs/<name>/stage3/stage4_task/`). To prepare one by hand, put everything for one task in one folder, with a config file next to the data:

```
my_task/
├── scene.xml          simulation scene (MuJoCo MJCF): the robot and the objects
├── assets/            meshes and textures the scene refers to
├── trajectory.npz     the verified Stage 3 trajectory (and the servo commands that produced it)
├── trajectory_manifest.json  the Stage 3 subgoals and what each one must achieve, for the physics replay
├── human_video.mp4    the original human video
├── sim_video.mp4      optional: the Stage 3 simulation video, used only in a comparison video
└── stage4.json        the config: cameras, what to randomize, how many samples
```

| Input | Where it goes | Notes |
|---|---|---|
| Simulation | `scene.xml` | The scene Stage 2 reconstructed, with the robot from Stage 3, as MJCF with its meshes and textures next to it. |
| Robot | inside `scene.xml` | Name the body of each end effector in `robot.end_effectors`. |
| Trajectory | `trajectory.npz` | `qpos` of shape `[T, nq]`, `fps`, and `labels`: one phase name per frame. Optional `reference` `[T, nq]`: the hand servo commands Stage 3 executed (fingers commanded past the object's surface, so they press); the physics replay uses them. |
| Subgoal expectations | `trajectory_manifest.json` | Stage 3's manifest: per subgoal its frame window and `expect` (rise, in hand, at target, released, supported on, untouched ...). The physics replay re-checks them on every new layout. |
| Human video | `human_video.mp4` | The camera must not move during the clip. |
| Ego camera | `cameras.ego` | Position, orientation and vertical field of view of the camera that filmed the human video, in the scene's world frame. |
| New views | `cameras.views` | Extra cameras, each attached to a body: for a wrist camera, its pose relative to the hand and its field of view. |
| Number of samples | `num_samples` | How many randomized episodes to keep. |

## Run

```bash
conda activate dexagent-stage4
cd stage4_robot_data_generation
python run_stage4.py /path/to/my_task/stage4.json
```

That runs all five steps (six with `lerobot.enabled`). To run or re-run only some of them:

```bash
python run_stage4.py /path/to/my_task/stage4.json --steps randomize
python run_stage4.py /path/to/my_task/stage4.json --steps export,render,composite
```

## Outputs

Everything is written under `output_dir` (default `outputs/` next to the config; for a run of `run_dexagent.py` that is `runs/<name>/stage3/stage4_task/outputs/`):

```
outputs/
├── randomize.json               every layout that was tried, and why rejected ones failed
├── plate/
│   ├── plate.png                the clean background
│   └── preview.mp4              human video with the erase mask | VOID result
├── samples/
│   ├── original/                the un-randomized trajectory
│   │   ├── ego.mp4, wrist_left.mp4, wrist_right.mp4
│   │   ├── preview.mp4          all views side by side
│   │   ├── compare.mp4          human video | simulation | generated
│   │   ├── trajectory.npz       qpos, joint_names, labels, fps
│   │   └── sample.json          the layout and the check results
│   ├── 0000/                    first randomized episode, the same files (no compare.mp4) plus
│   │   └── physics_replay.npz   what the contact-physics replay of this layout did
│   └── ...
└── render/                      Blender scenes and raw frames (safe to delete afterwards)
```

## LeRobot export

The `lerobot` step turns the composited episodes into a LeRobot dataset (codebase version v2.1), the original first.
It runs with `--steps all` when `lerobot.enabled` is true, or on its own:

```bash
python run_stage4.py /path/to/my_task/stage4.json --steps lerobot
```

Settings (`lerobot` in `stage4.json`, or the `stage4.lerobot` block of the DexAgent YAML, which `run_dexagent.py` copies in):

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | include the step in `--steps all` |
| `fps` | `30` | dataset rate. The videos must be encoded at it: set `render.stride` to the trajectory rate (30 fps) divided by `fps`, e.g. `1` for 30 Hz |
| `output_dir` | `lerobot` | folder under `output_dir` |
| `repo_id` | `dexagent/<name>` | the dataset is written to `<output_dir>/<last part of repo_id>/` |
| `robot_type` | the trajectory's robot schema | `robot_type` in `meta/info.json` |
| `task` | Stage 1's task prompt | the language instruction of every episode |
| `include_original` | `true` | the un-randomized trajectory as episode 0 |
| `merge_coupled` | `true` | leave out joints that an equality constraint drives from another one (the second jaw of a one-motor gripper) |

Per frame:

| Feature | Content |
|---|---|
| `observation.state` | measured joint positions of the robot (rad; gripper jaws in m), in trajectory order |
| `action` | absolute joint position targets: the servo reference of the next frame. The jaws' targets are commanded past the object's surface while grasping, so a policy learns to squeeze |
| `observation.images.<view>` | the composited video of each rendered view (`ego`, then `cameras.views`), H.264 |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | as LeRobot defines them |

```
outputs/lerobot/<name>/
├── meta/info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl
├── meta/stats.json            pooled statistics (LeRobot v2.1 aggregates episodes_stats.jsonl itself; GR00T reads this)
├── meta/dexagent_sources.json which samples/<name> each episode came from
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/observation.images.ego/episode_000000.mp4
```

Load it with `LeRobotDataset(repo_id, root=<that folder>)` (LeRobot 0.3.x, the last releases on v2.1). For GR00T, add
the `meta/modality.json` its data loader asks for, which names the slices of the state and action vectors
(`meta/info.json` lists the joint names in order).

### Re-rendering a finished run at another rate

The randomized layouts and the clean plate do not depend on the frame rate, so a run rendered at one stride can be
re-rendered at another (here every frame, 30 fps, and exported) without drawing new layouts or running VOID again:

```bash
conda activate dexagent-stage4
cd stage4_robot_data_generation
T=../runs/<name>/stage3/stage4_task/stage4.json
python - "$T" <<'EOF'
import json, sys; p = sys.argv[1]; c = json.load(open(p))
c.setdefault("render", {})["stride"] = 1
c["lerobot"] = {"enabled": True, "fps": 30}
json.dump(c, open(p, "w"), indent=1)
EOF
python run_stage4.py $T --steps export,render,composite,lerobot
```

Rendering every frame takes twice as long as every second frame. A later `run_dexagent.py --from stage4` starts again
from a fresh copy of the task folder and takes these settings from the `stage4` block of the YAML instead
(`render: {stride: 1}`, `lerobot: {enabled: true, fps: 30}`).

## Inspecting the episodes

Each accepted episode is a folder `outputs/samples/<id>/` (`original`, `0000`, `0001`, ...):

| File | What it shows |
|---|---|
| `preview.mp4` | all views side by side, as composited; the quickest look |
| `compare.mp4` (`original` only) | the human video, the Stage 3 simulation and the generated video side by side |
| `trajectory.npz` | `qpos` per frame (the physics replay of this layout), `reference` (the servo targets), `labels` (phase per frame), `joint_names`, `fps` |
| `scene_physics.xml` | the scene with this layout's object poses: open it with `trajectory.npz` to replay the episode |
| `sample.json` | the layout (`layout.objects`: dx, dy in m and dyaw in deg per object; `layout.robot`: start offset per arm joint in rad) and every check's numbers |

Replay an episode in the interactive MuJoCo viewer (space pauses, arrow keys step a frame, backspace restarts; the
phase is printed as it changes):

```bash
python stage3_trajectory_generation/view_trajectory.py runs/<name>/stage3/stage4_task/outputs/samples/0000
python stage3_trajectory_generation/view_trajectory.py runs/<name>/stage3/stage4_task/outputs/samples/0000 --speed 0.5 --loop
```

`outputs/randomize.json` lists every layout that was drawn, in order; the accepted ones are the samples `0000`,
`0001`, ... in the same order. To see why the others were rejected:

```bash
python -c "
import json
r = json.load(open('runs/<name>/stage3/stage4_task/outputs/randomize.json'))
print(f\"{r['accepted']} of {r['wanted']} wanted, {r['attempts']} draws\")
for row in r['rows']:
    print(row['attempt'], 'OK  ' if row['ok'] else 'FAIL', row.get('method'),
          {k: [round(x, 3) for x in v] for k, v in row['layout'].items()}, row['failed'] or '')
"
```

`method` says how the trajectory was adapted to the layout: `warp` (the Stage 3 motion bent to the new object pose),
`warp+retime` (also slowed down where it got too fast) or `reexecuted` (the Stage 3 plan run again on the layout,
when `reexecute` is on and the warp failed); `(no start offset)` means it only passed without the arms' random start offset.

## Runtime

Measured on one RTX A6000 for the shipped cup example: a 351-frame trajectory rendered at every second frame (176 frames per episode), three views at 1280×1024, 32 samples per pixel, 4 render processes.

| Step | Time |
|---|---|
| `randomize` | about 1.7 s per draw for the kinematic checks, plus the physics replay |
| physics replay | one replay of the episode for the original, plus one per kinematically accepted layout (10 s for the cup, 23 s for the 1,630-frame computer installation) |
| `plate` | 6 min, once per task (SAM 3 65 s, VOID 190 s); 3 to 7 min on the other tasks |
| `export` | 33 s per episode |
| `render` | 8 min per episode |
| `composite` | 44 s per episode |

The whole example, three randomized episodes plus the original, ran in 43 minutes. Rendering dominates, and it scales with the number of episodes, views and frames. Raise `render.stride` or lower `render.samples` to trade quality for time. Renders are cached: re-running `render` skips frames that are already on disk.

## Third-party components

Stage 4 builds on [VOID](https://github.com/netflix/void-model) with its [CogVideoX-Fun](https://github.com/aigc-apps/CogVideoX-Fun) base model for video object removal, [SAM 3](https://github.com/facebookresearch/sam3) for text-prompted masks, [MuJoCo](https://mujoco.org) for kinematics, and [Blender](https://www.blender.org) Cycles for rendering. Each has its own license and terms; please check them before use.
