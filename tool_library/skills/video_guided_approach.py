"""Grasp and approach candidates shaped like the demonstrator's reach: a grasp pose turned about the object's vertical
axis to where the human hand settled, approached along the direction the hand came from."""
from __future__ import annotations

TOOL = {
    "name": "video_guided_approach",
    "title": "Video-guided grasp approach",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "From a human_hand_track and a grasp pose on an upright object: the pose turned about the object's vertical "
               "axis so the hand sits on the side of the object where the human hand settled, with a pre-grasp out "
               "along the direction the hand came in from and a final straight run along the gripper's own axis; "
               "ranked by closeness to the video and, with bench_scene, screened for the arm (torso included).",
    "description": (
        "The human hand's settled centroid gives the side of the object it holds (its azimuth about the object's "
        "vertical axis); the pose (any grasp tool's format, with its recorded `approach`) is turned about that axis "
        "so its hand sits at that azimuth, then by `yaw_step_deg` steps on both sides up to `yaw_span_deg` (the arm "
        "may not hold the exact one). Each candidate gets `approach` = the unit blend of the video's incoming "
        "direction (entry -> settle, tilted up by `incoming_rise`) and the gripper's own approach axis, "
        "`approach_waypoints` = [final_m along the gripper axis] and the plan entry's `approach_m`: the executor "
        "reaches the pre-grasp approach_m out along the blend, moves in to final_m in front of the grasp and slides "
        "straight on, so the visible approach comes from the side the human hand came from while the jaws still "
        "enter along their own axis. A turn changes the contact on a non-round object: only round, upright parts "
        "(a cup, a bottle) keep their grasp under it; re-run the grasp stress tests on the chosen candidate. With "
        "bench_scene each candidate is screened by armhand.screen_poses (exact IK, the arm probe and the torso) at the "
        "pre-grasp, the grasp and `arm_waypoints`; for a carry to a destination near the robot's body, pass the "
        "destination there, and consider a turn of the carried object about the vertical (the plan's carry `turn`), "
        "which a round object does not show."
    ),
    "inputs": [
        {"name": "track_json", "type": "path | dict", "description": "human_hand_track's track.json."},
        {"name": "pose", "type": "dict | path", "description": "The grasp pose to shape (pos, euler, approach, finger targets), a list of poses (the first is used) or a grasp tool's out_json (its first `grasps` entry), e.g. from optimize_grasp_pose or generate_parallel_jaw_grasps."},
        {"name": "object_pos", "type": "list[3]", "description": "The object's vertical axis: its centre on the table (world)."},
        {"name": "side", "type": "str", "description": "Station of the grasp. Default 'right'."},
        {"name": "approach_m, final_m", "type": "float", "description": "Pre-grasp distance along the blended direction and the straight final run along the gripper axis. Defaults 0.12, 0.05."},
        {"name": "incoming_rise", "type": "float", "description": "Upward component added to the (horizontal) incoming direction before normalising. Default 0.25."},
        {"name": "yaw_span_deg, yaw_step_deg", "type": "float", "description": "Candidates around the video's azimuth. Defaults 30, 5."},
        {"name": "bench_scene", "type": "path | None", "description": "Stage 2 scene.xml: screen every candidate for the station's arm."},
        {"name": "arm_waypoints", "type": "list | None", "description": "As optimize_grasp_pose's (held-object targets through the carry and the destination)."},
        {"name": "out_json", "type": "path | None", "description": "Write the result here."},
    ],
    "outputs": [
        {"name": "video", "type": "dict", "description": "incoming direction, settled palm azimuth about the object (deg), entry_s, settle_s, reach_s (entry to settle)."},
        {"name": "candidates", "type": "list", "description": "[{yaw_deg (turn applied to the input pose), video_offset_deg, pose, screen}] ranked: screened ok first, then by |video_offset_deg|."},
        {"name": "pose_candidates", "type": "list", "description": "The ranked poses, for execute_subgoal_plan's pose_candidates (the grasp entry also needs approach_m)."},
        {"name": "approach_m", "type": "float", "description": "Put this in the grasp subgoal's plan entry."},
    ],
    "status": "tested",
    "entrypoint": "video_guided_approach",
    "requirements": ["mujoco", "numpy", "scipy"],
    "tags": ["grasp", "approach", "video", "human", "reach"],
    "used_in": ["Pick up the cup and place it on the plate"],
    "cli": "python -m tool_library.skills.video_guided_approach TRACK_JSON POSE_JSON X Y Z",
}

import json
import sys
from pathlib import Path

import numpy as np


def _load(x):
    if isinstance(x, (str, Path)):
        x = json.loads(Path(x).read_text())
    return x


def video_direction(track):
    """(unit horizontal direction the hand came in from, settled centroid xy, entry_s, settle_s)."""
    pts = track["track"]
    if len(pts) < 2:
        raise ValueError("the track needs at least two frames with a hand")
    settle = track.get("settle_s")
    end = next((p for p in pts if settle is not None and p["t"] >= settle), pts[-1])
    c0, c1 = np.asarray(pts[0]["centroid"][:2], float), np.asarray(end["centroid"][:2], float)
    v = c0 - c1
    if np.linalg.norm(v) < 1e-3:
        raise ValueError("the hand does not move in the track: widen time_window_s")
    return v / np.linalg.norm(v), c1, track.get("entry_s"), settle


def turned_pose(pose, axis_xy, yaw_deg):
    """The pose turned by yaw_deg about the vertical axis through axis_xy (hand position, orientation, approach)."""
    from scipy.spatial.transform import Rotation as Rot
    from tool_library.common import trajectory as TJ
    Rz = Rot.from_euler("z", float(yaw_deg), degrees=True).as_matrix()
    c = np.array([axis_xy[0], axis_xy[1], 0.0])
    p = dict(pose)
    p["pos"] = (c + Rz @ (np.asarray(pose["pos"], float) - c)).tolist()
    p["euler"] = [float(v) for v in TJ.mat_to_euler(Rz @ TJ.euler_to_mat(np.asarray(pose["euler"], float)))]
    for k in ("approach",):
        if pose.get(k) is not None:
            p[k] = (Rz @ np.asarray(pose[k], float)).tolist()
    for k in ("points", "targets"):
        if isinstance(pose.get(k), dict):
            p[k] = {f: (c + Rz @ (np.asarray(v, float) - c)).tolist() for f, v in pose[k].items()}
    if isinstance(pose.get("normals"), dict):
        p["normals"] = {f: (Rz @ np.asarray(v, float)).tolist() for f, v in pose["normals"].items()}
    return p


def _azimuth(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _wrap(a):
    return float((a + 180.0) % 360.0 - 180.0)


def video_guided_approach(track_json, pose, object_pos, side="right", approach_m=0.12, final_m=0.05, incoming_rise=0.25,
                          yaw_span_deg=30.0, yaw_step_deg=5.0, bench_scene=None, arm_waypoints=None, out_json=None):
    from tool_library.common import plan as P, trajectory as TJ
    track, pose = _load(track_json), _load(pose)
    if isinstance(pose, dict) and "grasps" in pose and "pos" not in pose:
        if not pose["grasps"]:
            raise ValueError("the grasp file has no grasps")
        pose = pose["grasps"][0]
    if isinstance(pose, list):
        pose = pose[0]
    if isinstance(pose, dict) and "pose" in pose and "pos" not in pose:
        pose = pose["pose"]
    obj = np.asarray(object_pos, float)
    inc_xy, settled, entry_s, settle_s = video_direction(track)
    hand_az = _azimuth(settled - obj[:2])
    pose_az = _azimuth(np.asarray(pose["pos"][:2], float) - obj[:2])
    base = _wrap(hand_az - pose_az)
    n = int(np.floor(float(yaw_span_deg) / max(float(yaw_step_deg), 1e-6)))
    offsets = [0.0] + [s * k * float(yaw_step_deg) for k in range(1, n + 1) for s in (1, -1)]
    inc = np.r_[inc_xy, float(incoming_rise)]
    inc /= np.linalg.norm(inc)
    cands = []
    for off in offsets:
        yaw = _wrap(base + off)
        p = turned_pose(pose, obj[:2], yaw)
        ga = P.retreat_axis(p, TJ.euler_to_mat(np.asarray(p["euler"], float)))
        b = inc + ga
        b = b / np.linalg.norm(b) if np.linalg.norm(b) > 1e-6 else ga
        p["approach"] = [float(v) for v in b]
        p["approach_waypoints"] = [[float(v) for v in final_m * ga]]
        p["video_guided"] = {"yaw_deg": round(yaw, 2), "video_offset_deg": round(off, 2), "gripper_axis": [round(float(v), 4) for v in ga]}
        cands.append({"yaw_deg": round(yaw, 2), "video_offset_deg": round(off, 2), "pose": p, "screen": None})
    if bench_scene:
        import mujoco
        from tool_library.common import armhand as AH, episode as E
        bm, _ = E.build_execution_model(str(bench_scene))
        q = bm.key_qpos[0].copy() if bm.nkey else bm.qpos0.copy()
        probe = AH.ArmProbe(str(bench_scene))
        scr = AH.screen_poses(bm, side, [c["pose"] for c in cands], q, probe=probe, approach_m=float(approach_m), waypoints=arm_waypoints)
        for c, s in zip(cands, scr):
            c["screen"] = {k: s.get(k) for k in ("ok", "reason", "failed_at", "residual_mm", "residual_deg")}
    cands.sort(key=lambda c: (0 if (c["screen"] or {}).get("ok", True) else 1, abs(c["video_offset_deg"])))
    res = {"video": {"incoming_dir": [round(float(v), 4) for v in inc], "hand_azimuth_deg": round(hand_az, 2),
                     "pose_azimuth_deg": round(pose_az, 2), "entry_s": entry_s, "settle_s": settle_s,
                     "reach_s": None if entry_s is None or settle_s is None else round(settle_s - entry_s, 3)},
           "candidates": cands, "pose_candidates": [c["pose"] for c in cands], "approach_m": float(approach_m),
           "n_screen_ok": sum(1 for c in cands if (c["screen"] or {}).get("ok")) if bench_scene else None}
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(res, indent=1))
        res["out_json"] = str(out_json)
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("track_json"); ap.add_argument("pose_json"); ap.add_argument("x", type=float); ap.add_argument("y", type=float); ap.add_argument("z", type=float)
    ap.add_argument("--bench-scene"); ap.add_argument("--out-json")
    a = ap.parse_args(argv)
    r = video_guided_approach(a.track_json, a.pose_json, [a.x, a.y, a.z], bench_scene=a.bench_scene, out_json=a.out_json)
    print(json.dumps({"video": r["video"], "ranked": [(c["yaw_deg"], c["video_offset_deg"], c["screen"]) for c in r["candidates"]]}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
