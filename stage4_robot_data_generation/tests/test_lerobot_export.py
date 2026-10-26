"""The LeRobot v2.1 export (stage4/lerobot.py) on a synthetic two-episode run.

A one-arm robot (one hinge, two jaws coupled by an equality constraint) and a free object; trajectory.npz and two
composited 30 fps views per sample. Checks:
  - state and action hold the robot's joints only, the coupled second jaw left out;
  - the action of frame t is the servo reference of frame t + 1 (the last frame repeats its own);
  - the original is episode 0 and the frame index runs on across episodes;
  - parquet columns are fixed-size float32 lists with the `Sequence` feature metadata datasets 3.x reads;
  - meta/info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl and stats.json agree with the data;
  - a dataset rate the videos are not encoded at is refused.

    cd stage4_robot_data_generation && python tests/test_lerobot_export.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from stage4 import lerobot  # noqa: E402
from stage4.config import load_config  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="stage4_lerobot_", dir=os.environ.get("TMPDIR")))
FAILS = 0

SCENE = """<mujoco>
  <worldbody>
    <body name="robot_right_arm_link">
      <joint name="robot_right_arm_j1_joint" type="hinge" axis="0 0 1"/>
      <geom type="box" size=".02 .02 .02"/>
      <body name="robot_right_right_thumb">
        <joint name="robot_right_right_thumb_jaw" type="slide" axis="0 1 0" range="-.02 .02"/>
        <geom type="box" size=".005 .005 .005"/>
      </body>
      <body name="robot_right_right_index">
        <joint name="robot_right_right_index_jaw" type="slide" axis="0 -1 0" range="-.02 .02"/>
        <geom type="box" size=".005 .005 .005"/>
      </body>
    </body>
    <body name="cup" pos=".3 0 .05"><freejoint name="cup_free"/><geom type="cylinder" size=".03 .05"/></body>
  </worldbody>
  <equality><joint name="jaw_coupling" joint1="robot_right_right_index_jaw" joint2="robot_right_right_thumb_jaw"/></equality>
</mujoco>
"""
JOINTS = ["robot_right_arm_j1_joint", "robot_right_right_thumb_jaw", "robot_right_right_index_jaw"] + \
    [f"cup_free[{i}]" for i in range(7)]


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    FAILS += 0 if ok else 1


def video(path, n, fps, shade):
    w = imageio.get_writer(str(path), fps=fps, macro_block_size=1, codec="libx264")
    for i in range(n):
        f = np.full((48, 64, 3), shade, np.uint8)
        f[:, : (i % 64)] = 255 - shade
        w.append_data(f)
    w.close()


def sample(out, name, T, seed, fps=30):
    d = out / "samples" / name
    d.mkdir(parents=True)
    (d / "scene_physics.xml").write_text(SCENE)
    rng = np.random.default_rng(seed)
    q = np.zeros((T, len(JOINTS)))
    q[:, 0] = np.linspace(0, 1, T) + seed
    q[:, 1] = q[:, 2] = np.linspace(0, 0.01, T)
    q[:, 3:6] = [0.3, 0, 0.05]
    q[:, 6] = 1.0
    ref = q.copy()
    ref[:, 1] = ref[:, 2] = 0.02                      # the jaws commanded past the object
    ref[:, 0] += rng.normal(0, 1e-3, T)
    np.savez(d / "trajectory.npz", qpos=q, reference=ref, fps=np.float64(30.0), joint_names=np.array(JOINTS),
             robot_schema=np.array("test_robot"), labels=np.array(["phase"] * T))
    for v, shade in (("ego", 40), ("wrist_right", 90)):
        video(d / f"{v}.mp4", T, fps, shade)
    return q, ref


def main():
    (TMP / "scene.xml").write_text(SCENE)
    np.savez(TMP / "trajectory.npz", qpos=np.zeros((1, len(JOINTS))), fps=np.float64(30.0))
    video(TMP / "human_video.mp4", 4, 30, 0)
    (TMP / "semantics.json").write_text(json.dumps({"task_prompt": "Put the cup on the plate"}))
    (TMP / "stage4.json").write_text(json.dumps({
        "name": "synthetic", "scene": "scene.xml", "trajectory": "trajectory.npz", "human_video": "human_video.mp4",
        "robot": {"end_effectors": {"right": "robot_right_arm_link"}},
        "cameras": {"width": 64, "height": 48, "ego": {"pos": [0, 0, 1], "quat_wxyz": [1, 0, 0, 0], "fovy": 40},
                    "views": {"wrist_right": {"body": "robot_right_arm_link", "pos": [0, 0, 0.05],
                                              "quat_wxyz": [1, 0, 0, 0], "fovy": 60}}},
        "table": {"top_z": 0.0, "center": [0.0, 0.3], "half_size": [0.5, 0.4]},
        "goal": {"semantics": "semantics.json"}}))
    cfg = load_config(TMP / "stage4.json")
    qo, ro = sample(cfg.out, "original", 40, 0)
    q1, r1 = sample(cfg.out, "0000", 50, 1)

    cfg["lerobot"] = {"fps": 15}
    try:
        lerobot.run(cfg, log=lambda m: None)
        check("a rate the videos are not encoded at is refused", False)
    except AssertionError as e:
        check("a rate the videos are not encoded at is refused", "render.stride" in str(e), str(e)[:120])

    cfg["lerobot"] = {"fps": 30}
    lerobot.run(cfg, log=lambda m: None)
    root = cfg.out / "lerobot" / "synthetic"
    info = json.loads((root / "meta/info.json").read_text())
    names = info["features"]["observation.state"]["names"]
    check("state holds the robot joints, the coupled jaw left out", names == ["right_arm_j1_joint", "right_thumb_jaw"], str(names))
    check("info totals", info["codebase_version"] == "v2.1" and info["total_episodes"] == 2 and info["total_frames"] == 90
          and info["total_videos"] == 4 and info["fps"] == 30 and info["robot_type"] == "test_robot", json.dumps(info)[:200])
    check("video features", set(k for k, f in info["features"].items() if f["dtype"] == "video")
          == {"observation.images.ego", "observation.images.wrist_right"}
          and info["features"]["observation.images.ego"]["shape"] == [48, 64, 3])

    t0 = pq.read_table(root / "data/chunk-000/episode_000000.parquet")
    t1 = pq.read_table(root / "data/chunk-000/episode_000001.parquet")
    meta = json.loads(t0.schema.metadata[b"huggingface"])["info"]["features"]
    at = t0.schema.field("action").type
    check("parquet vectors are fixed-size float32 lists with Sequence metadata",
          pa.types.is_fixed_size_list(at) and at.list_size == 2 and at.value_type == pa.float32()
          and meta["action"]["_type"] == "Sequence"
          and meta["action"]["length"] == 2, str(t0.schema.field("action").type))
    s0, a0 = np.array(t0.column("observation.state").to_pylist()), np.array(t0.column("action").to_pylist())
    check("episode 0 is the original", len(s0) == 40 and np.allclose(s0, qo[:, :2], atol=1e-6))
    check("action of frame t is the reference of frame t + 1", np.allclose(a0[:-1], ro[1:, :2], atol=1e-6)
          and np.allclose(a0[-1], ro[-1, :2], atol=1e-6))
    check("the jaw action is the squeezing command, not the measured jaw", np.allclose(a0[:, 1], 0.02, atol=1e-6))
    idx = t1.column("index").to_pylist()
    check("frame index runs on across episodes", idx[0] == 40 and idx[-1] == 89 and t1.column("episode_index").to_pylist()[0] == 1)
    check("timestamps at 30 fps", np.allclose(t1.column("timestamp").to_pylist(), np.arange(50) / 30.0, atol=1e-6))

    eps = [json.loads(l) for l in (root / "meta/episodes.jsonl").read_text().splitlines()]
    tasks = [json.loads(l) for l in (root / "meta/tasks.jsonl").read_text().splitlines()]
    est = [json.loads(l) for l in (root / "meta/episodes_stats.jsonl").read_text().splitlines()]
    check("episodes and task", [e["length"] for e in eps] == [40, 50] and tasks == [{"task_index": 0, "task": "Put the cup on the plate"}]
          and eps[0]["tasks"] == ["Put the cup on the plate"])
    img = est[0]["stats"]["observation.images.ego"]
    check("per-episode stats", est[1]["stats"]["observation.state"]["count"] == [50]
          and np.allclose(est[1]["stats"]["observation.state"]["mean"], q1[:, :2].mean(0), atol=1e-5)
          and np.array(img["mean"]).shape == (3, 1, 1) and 0 <= np.min(img["min"]) and np.max(img["max"]) <= 1)
    agg = json.loads((root / "meta/stats.json").read_text())
    both = np.concatenate([qo[:, :2], q1[:, :2]])
    check("aggregate stats pool the episodes", np.allclose(agg["observation.state"]["mean"], both.mean(0), atol=1e-5)
          and np.allclose(agg["observation.state"]["std"], both.std(0), atol=1e-5))
    check("videos copied per episode and view",
          all((root / f"videos/chunk-000/observation.images.{v}/episode_{e:06d}.mp4").exists()
              for v in ("ego", "wrist_right") for e in (0, 1)))
    print(f"{'ALL PASS' if FAILS == 0 else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
