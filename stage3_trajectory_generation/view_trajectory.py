"""Replay a recorded episode in the interactive MuJoCo viewer.

    python stage3_trajectory_generation/view_trajectory.py <episode dir>
    python stage3_trajectory_generation/view_trajectory.py --scene scene_physics.xml --trajectory trajectory.npz

The episode dir is an execute_subgoal_plan out_dir (scene_physics.xml + trajectory.npz), e.g. a Stage 3 candidate
under runs/<name>/stage3/.candidates/<id>/. Keys: space pause / play, right / left arrow one frame forward / back
(while paused), backspace restart. The phase label of each frame is printed when it changes.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", nargs="?", type=Path, help="folder holding scene_physics.xml and trajectory.npz")
    ap.add_argument("--scene", type=Path)
    ap.add_argument("--trajectory", type=Path)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (default 1 = real time)")
    ap.add_argument("--substeps", action="store_true", help="replay every physics substep instead of the frames")
    ap.add_argument("--loop", action="store_true", help="start over at the end")
    args = ap.parse_args()
    scene = args.scene or (args.episode / "scene_physics.xml")
    traj = args.trajectory or (args.episode / "trajectory.npz")
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    z = np.load(traj, allow_pickle=True)
    if args.substeps and "substep_qpos" in z:
        Q, T = z["substep_qpos"], z["substep_time_s"]
        labels = None
    else:
        Q = z["qpos"]
        T = z["time_s"] if "time_s" in z else np.arange(len(Q)) / float(z["fps"] if "fps" in z else 30.0)
        labels = [str(x) for x in z["labels"]] if "labels" in z else None
    if Q.shape[1] != m.nq:
        raise SystemExit(f"{traj} has {Q.shape[1]} qpos columns but {scene} has nq {m.nq}: use the episode's own "
                         f"scene_physics.xml")
    state = {"k": 0, "paused": False, "step": 0, "restart": False}

    def key(code):
        if code == 32:
            state["paused"] = not state["paused"]
        elif code == 262:
            state["step"] += 1
        elif code == 263:
            state["step"] -= 1
        elif code == 259:
            state["restart"] = True

    print(f"{len(Q)} {'substeps' if labels is None else 'frames'}, {T[-1] - T[0]:.1f} s")
    shown = None
    with mujoco.viewer.launch_passive(m, d, key_callback=key) as v:
        t0 = time.time() - T[0] / args.speed
        while v.is_running():
            if state["restart"]:
                state.update(k=0, restart=False)
                t0 = time.time() - T[0] / args.speed
            if state["paused"]:
                state["k"] = int(np.clip(state["k"] + state["step"], 0, len(Q) - 1))
                state["step"] = 0
                t0 = time.time() - T[state["k"]] / args.speed
            else:
                k = int(np.searchsorted(T, (time.time() - t0) * args.speed, side="right")) - 1
                if k >= len(Q) - 1 and args.loop:
                    k, t0 = 0, time.time() - T[0] / args.speed
                state["k"] = int(np.clip(k, 0, len(Q) - 1))
            k = state["k"]
            d.qpos[:] = Q[k]
            mujoco.mj_forward(m, d)
            if labels is not None and labels[k] != shown:
                shown = labels[k]
                print(f"[{T[k]:6.2f} s, frame {k}] {shown}")
            v.sync()
            time.sleep(1.0 / 120.0)


if __name__ == "__main__":
    main()
