# Stage 2: Property-Based Simulation Reconstruction

The video and Stage 1's `scene_semantics.json` in; a simulation scene out, with every task object rebuilt with the properties the task needs (joints that move, threads that turn, cords that bend) and placed where the video shows it, seen through the video's own camera, on the robot setup.

## Run

```bash
python stage2_scene_reconstruction/run_stage2.py dexagent.yaml                 # after Stage 1 of the same run
python stage2_scene_reconstruction/run_stage2.py dexagent.yaml --egodex ep.hdf5 --name egodex_bottle   # an EgoDex episode: ep.mp4 beside it is the video
```

Needs: the model under `agent:` in `dexagent.yaml`, a GPU for SAM 3 (masks), and, for deformable objects, Isaac Sim 5.0 in its own Python (`environments.isaac`). Perception models run in the interpreters named under `environments` / `tool_environments`.

Output, under `<run.output_dir>/<run.name>/stage2/`:

```
scene.xml              the setup scene with the reconstructed objects at their fitted poses; camera `video_cam` is the video camera
camera.json            the video camera in the rig frame
masks/                 one SAM 3 mask per object and masks/hands.png
assets/<id>/           each object's asset as built by its reconstruction skill (MJCF + meshes + params)
isaac/                 for deformable objects: the Isaac Sim stage, its render and the settling report
frame0_compare.png     video frame | simulation | blend (| Isaac | blend)
scene_recon.json       the manifest: per object the skill, parameters, pose, mask, IoU, verification plan, verifier reports, property coverage
trace.jsonl            every model turn, tool call and verifier verdict
new_tools.json         skills or verifiers the agent added to the library during the run, if any
```

## View the scene

Open `scene.xml` in the MuJoCo viewer at the `home` keyframe (the robot at rest, the objects as frame 0 shows them):

```bash
python -c "
import sys, mujoco, mujoco.viewer
m = mujoco.MjModel.from_xml_path(sys.argv[1]); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.viewer.launch(m, d)" runs/<name>/stage2/scene.xml
```

Space pauses or runs the physics (the objects should stay put: they rest where the video shows them). To look
through the video camera, pick `video_cam` under Rendering > Camera in the left panel; `frame0_compare.png` puts the
same view beside the video frame. Simulation > Key > Load key returns to `home`. `python -m mujoco.viewer
--mjcf=runs/<name>/stage2/scene.xml` also opens it, but at the joints' zero values instead of `home`.
