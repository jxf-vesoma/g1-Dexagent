"""Step 6 (optional): write the composited episodes as a LeRobot v2.1 dataset.

    outputs/samples/<sample>/{trajectory.npz, <view>.mp4} --> outputs/lerobot/<repo>/

One sample is one episode. Per frame:
    observation.state          measured robot joint positions (rad; jaws in m), qpos of the recorded episode
    action                     absolute joint position targets: the servo reference of the next frame (`reference` in
                               trajectory.npz is the joint path the arm and jaw servos tracked, jaws commanded past the
                               object's surface so they squeeze); the last frame repeats its own target. With
                               lerobot.grip_cap, a gripper's target is at most `margin` past its measured position
                               of that frame in the closing direction
    observation.images.<view>  the composited video of every rendered view (ego first), copied as encoded

The sim jaw servo holds an object by a force-limited command to fully closed. A position-controlled real gripper given
that target crushes the object, so grip_cap labels a grip as the squeeze it can execute: contact plus a margin.

A joint that an equality constraint drives from another (the second jaw of a one-motor gripper) is left out of both
vectors unless lerobot.merge_coupled is false. The language instruction is lerobot.task, or Stage 1's task prompt.

Layout (LeRobot codebase v2.1): meta/{info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl, stats.json},
data/chunk-000/episode_000000.parquet, videos/chunk-000/observation.images.<view>/episode_000000.mp4. meta/stats.json
(the aggregate) is not read by LeRobot v2.1, which aggregates episodes_stats.jsonl itself; GR00T's loader reads it.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np

CHUNK = 1000
DEFAULTS = {
    "enabled": False,            # run this step as part of `--steps all`
    "fps": 30,                   # dataset rate; the videos must be encoded at it (render.stride = trajectory fps / fps)
    "output_dir": "lerobot",     # under the Stage 4 output_dir
    "repo_id": None,             # default: dexagent/<config name>
    "robot_type": None,          # default: the trajectory's robot_schema
    "task": None,                # default: Stage 1's task_prompt (goal.semantics)
    "include_original": True,    # the un-randomized trajectory as episode 0
    "merge_coupled": True,       # leave out joints an equality constraint drives from another joint
    "image_stats_frames": 100,   # frames per video sampled for the image statistics
    "grip_cap": None,            # {"joints": [feature-name globs], "margin": m or rad, "closing": +1 or -1}
}


def cap_grip(act, q, names, cap):
    """Gripper targets no further than cap["margin"] past the measured position, in the closing direction."""
    import fnmatch
    out = act.copy()
    s = float(cap.get("closing", 1.0))
    q_next = np.concatenate([q[1:], q[-1:]])
    for k, n in enumerate(names):
        if any(fnmatch.fnmatchcase(n, p) for p in cap["joints"]):
            past = np.minimum(s * (act[:, k] - q_next[:, k]), float(cap["margin"]))
            out[:, k] = q_next[:, k] + s * past
    return out


def settings(cfg):
    return {**DEFAULTS, **(cfg.get("lerobot") or {})}


def robot_columns(model, joint_names, prefix, merge_coupled):
    """(qpos columns, feature names) of the robot's hinge and slide joints, in trajectory order."""
    followers = set()
    if merge_coupled:
        for e in range(model.neq):
            if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[e] >= 0:
                followers.add(model.joint(int(model.eq_obj1id[e])).name)
    cols, names = [], []
    for c, n in enumerate(joint_names):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j < 0 or not n.startswith(prefix) or n in followers:
            continue
        if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        short = n[len(prefix):]
        for side in ("left", "right"):
            short = short.replace(f"{side}_{side}_", f"{side}_")
        cols.append(c)
        names.append(short)
    return cols, names


def video_info(path):
    r = imageio.get_reader(str(path))
    meta = r.get_meta_data()
    n = r.count_frames()
    r.close()
    w, h = meta["size"]
    return {"fps": float(meta["fps"]), "n": int(n), "width": int(w), "height": int(h),
            "codec": str(meta.get("codec", "h264")), "pix_fmt": str(meta.get("pix_fmt", "yuv420p")).split("(")[0]}


def vector_stats(x):
    x = np.asarray(x, np.float64).reshape(len(x), -1)
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(), "count": [int(len(x))]}


def image_stats(path, n_frames, k):
    """LeRobot's per-channel image statistics, values in [0, 1], shape (3, 1, 1), from k evenly spaced frames."""
    idx = set(np.unique(np.linspace(0, n_frames - 1, min(k, n_frames)).round().astype(int)).tolist())
    px = []
    r = imageio.get_reader(str(path))
    for i, f in enumerate(r):
        if i in idx:
            h, w = f.shape[:2]
            s = min(1.0, 150.0 / max(h, w))
            f = cv2.resize(f, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
            px.append(f.reshape(-1, 3).astype(np.float64) / 255.0)
    r.close()
    px = np.concatenate(px)
    shape = lambda v: [[[float(c)]] for c in v]
    return {"min": shape(px.min(0)), "max": shape(px.max(0)), "mean": shape(px.mean(0)), "std": shape(px.std(0)),
            "count": [len(idx)]}


def write_parquet(path, rows, n_joints):
    """One episode's table as LeRobot v2.1 writes it: fixed-size float32 lists, and the Hugging Face feature metadata
    in the `Sequence` form that datasets 3.x (LeRobot v2.1, GR00T) reads; datasets 4 would write `List`."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    vec = pa.list_(pa.float32(), n_joints)
    cols = {"observation.state": pa.array(rows["observation.state"], vec), "action": pa.array(rows["action"], vec),
            "timestamp": pa.array(rows["timestamp"], pa.float32())}
    for k in ("frame_index", "episode_index", "index", "task_index"):
        cols[k] = pa.array(rows[k], pa.int64())
    seq = {"feature": {"dtype": "float32", "_type": "Value"}, "length": n_joints, "_type": "Sequence"}
    feats = {"observation.state": seq, "action": seq, "timestamp": {"dtype": "float32", "_type": "Value"},
             **{k: {"dtype": "int64", "_type": "Value"} for k in ("frame_index", "episode_index", "index", "task_index")}}
    table = pa.table(cols).replace_schema_metadata({"huggingface": json.dumps({"info": {"features": feats}})})
    pq.write_table(table, str(path))


def task_text(cfg, st):
    if st["task"]:
        return str(st["task"])
    sem = (cfg.get("goal") or {}).get("semantics")
    if sem and cfg.path_of(sem).exists():
        j = json.loads(cfg.path_of(sem).read_text())
        if j.get("task_prompt"):
            return str(j["task_prompt"])
    return str(cfg.get("name", "task"))


def run(cfg, log=print):
    from .export import current_samples, stage_views

    st = settings(cfg)
    fps = float(st["fps"])
    views = stage_views(cfg)
    samples = [s for s in current_samples(cfg) if st["include_original"] or s.name != "original"]
    samples.sort(key=lambda s: (s.name != "original", s.name))
    assert samples, "no samples: run the randomize step first"
    missing = [f"{s.name}/{v}.mp4" for s in samples for v in views if not (s / f"{v}.mp4").exists()]
    assert not missing, f"no composited video {missing[:6]}: run the composite step first"

    repo = st["repo_id"] or f"dexagent/{cfg.get('name', 'task')}"
    root = cfg.out / st["output_dir"] / repo.split("/")[-1]
    shutil.rmtree(root, ignore_errors=True)
    (root / "meta").mkdir(parents=True)
    task = task_text(cfg, st)

    cols = names = None
    robot_type = st["robot_type"]
    episodes, ep_stats, sources, total, vid_meta = [], [], [], 0, {}
    for e, sdir in enumerate(samples):
        z = np.load(sdir / "trajectory.npz", allow_pickle=True)
        scene = sdir / "scene_physics.xml"
        model = mujoco.MjModel.from_xml_path(str(scene if scene.exists() else cfg.path("scene")))
        c, n = robot_columns(model, [str(x) for x in z["joint_names"]], cfg["robot"]["body_prefix"], st["merge_coupled"])
        if cols is None:
            cols, names = c, n
            robot_type = robot_type or (str(z["robot_schema"]) if "robot_schema" in z.files else "dexagent")
        assert names == n, f"{sdir.name}: robot joints differ from the first episode's ({n} vs {names})"
        traj_fps = float(z["fps"])
        step = traj_fps / fps
        assert abs(step - round(step)) < 1e-6 and round(step) >= 1, (
            f"lerobot.fps {fps:g} must divide the trajectory rate {traj_fps:g}")
        step = int(round(step))
        info = {v: video_info(sdir / f"{v}.mp4") for v in views}
        for v, vi in info.items():
            assert abs(vi["fps"] - fps) < 1e-3, (
                f"{sdir.name}/{v}.mp4 is encoded at {vi['fps']:g} fps, not lerobot.fps {fps:g}: set render.stride to "
                f"{traj_fps / fps:g} and re-run the export, render and composite steps")
        q = np.asarray(z["qpos"], np.float64)[::step][:, cols]
        ref = np.asarray(z["reference"] if "reference" in z.files else z["qpos"], np.float64)[::step][:, cols]
        T = min([len(q)] + [vi["n"] for vi in info.values()])
        assert all(abs(vi["n"] - len(q)) <= 1 for vi in info.values()), (
            f"{sdir.name}: {len(q)} trajectory frames at {fps:g} fps but videos of "
            f"{ {v: vi['n'] for v, vi in info.items()} } frames")
        q, ref = q[:T], ref[:T]
        act = np.concatenate([ref[1:], ref[-1:]])
        if st["grip_cap"]:
            act = cap_grip(act, q, names, st["grip_cap"])
        t = np.arange(T) / fps
        rows = {"observation.state": q.astype(np.float32).tolist(), "action": act.astype(np.float32).tolist(),
                "timestamp": t.astype(np.float32).tolist(), "frame_index": list(range(T)),
                "episode_index": [e] * T, "index": list(range(total, total + T)), "task_index": [0] * T}
        chunk = e // CHUNK
        pq = root / f"data/chunk-{chunk:03d}/episode_{e:06d}.parquet"
        pq.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(pq, rows, len(names))
        stats = {"observation.state": vector_stats(q), "action": vector_stats(act), "timestamp": vector_stats(t),
                 "frame_index": vector_stats(np.arange(T)), "episode_index": vector_stats(np.full(T, e)),
                 "index": vector_stats(np.arange(total, total + T)), "task_index": vector_stats(np.zeros(T))}
        for v in views:
            key = f"observation.images.{v}"
            dst = root / f"videos/chunk-{chunk:03d}/{key}/episode_{e:06d}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sdir / f"{v}.mp4", dst)
            stats[key] = image_stats(dst, info[v]["n"], int(st["image_stats_frames"]))
            vid_meta.setdefault(key, info[v])
        episodes.append({"episode_index": e, "tasks": [task], "length": T})
        ep_stats.append({"episode_index": e, "stats": stats})
        sources.append({"episode_index": e, "sample": sdir.name, "frames": T})
        total += T
        log(f"[lerobot] episode {e}: sample {sdir.name}, {T} frames")

    features = {
        "observation.state": {"dtype": "float32", "shape": [len(names)], "names": names},
        "action": {"dtype": "float32", "shape": [len(names)], "names": names},
    }
    fps_out = int(fps) if fps.is_integer() else fps
    for key, vi in vid_meta.items():
        features[key] = {"dtype": "video", "shape": [vi["height"], vi["width"], 3], "names": ["height", "width", "channels"],
                         "info": {"video.height": vi["height"], "video.width": vi["width"], "video.codec": vi["codec"],
                                  "video.pix_fmt": vi["pix_fmt"], "video.is_depth_map": False, "video.fps": fps_out,
                                  "video.channels": 3, "has_audio": False}}
    for k, dt in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"), ("index", "int64"),
                  ("task_index", "int64")):
        features[k] = {"dtype": dt, "shape": [1], "names": None}
    n_eps = len(episodes)
    info = {"codebase_version": "v2.1", "robot_type": robot_type, "total_episodes": n_eps, "total_frames": total,
            "total_tasks": 1, "total_videos": n_eps * len(views), "total_chunks": math.ceil(n_eps / CHUNK),
            "chunks_size": CHUNK, "fps": fps_out, "splits": {"train": f"0:{n_eps}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features}
    meta = root / "meta"
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    (meta / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in episodes))
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep_stats))
    (meta / "stats.json").write_text(json.dumps(aggregate(ep_stats), indent=1))
    (meta / "dexagent_sources.json").write_text(json.dumps(sources, indent=1))
    log(f"[lerobot] {n_eps} episodes, {total} frames at {fps:g} fps, state/action {len(names)} joints, "
        f"views {views} -> {root}")


def aggregate(ep_stats):
    """Pooled statistics over all episodes, weighted by each episode's count (LeRobot's aggregate_stats)."""
    out = {}
    for key in ep_stats[0]["stats"]:
        S = [s["stats"][key] for s in ep_stats]
        w = np.array([s["count"][0] for s in S], np.float64)
        mean = np.array([s["mean"] for s in S], np.float64)
        var = np.array([s["std"] for s in S], np.float64) ** 2
        wb = w.reshape((-1,) + (1,) * (mean.ndim - 1))
        m = (wb * mean).sum(0) / w.sum()
        v = (wb * (var + (mean - m) ** 2)).sum(0) / w.sum()
        out[key] = {"min": np.min([s["min"] for s in S], 0).tolist(), "max": np.max([s["max"] for s in S], 0).tolist(),
                    "mean": m.tolist(), "std": np.sqrt(v).tolist(), "count": [int(w.sum())]}
    return out
