"""The demonstrator's hand path on the table from the video: SAM 3 hand masks back-projected through the camera."""
from __future__ import annotations

TOOL = {
    "name": "human_hand_track",
    "title": "Human hand track from video",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "Segments the demonstrator's hand (SAM 3) in a time window of the video and back-projects each frame's hand "
               "centroid and leading point through the registered camera onto a horizontal plane: the hand's path on the "
               "bench in metres, when it enters the view and when it settles at the object.",
    "description": (
        "Decodes the video frames of `time_window_s` (every `stride`-th), segments `prompt` with SAM 3, keeps the "
        "hand's connected component (the largest one, or with two large ones the one on `side`'s half of the image), "
        "and back-projects its pixel centroid (plane `plane_z_m`) and its leading point (the mask pixel farthest from "
        "where the arm crosses the image border, plane `tip_z_m`) through camera.json. A monocular track has no depth "
        "of its own: the planes are assumptions (the palm about at the grasp height, the fingertips at the object's "
        "upper side), so the path's direction and timing are reliable, its height is not. `entry_s` is the first "
        "frame with a hand, `settle_s` the first frame after which the centroid moves less than `settle_speed_m_s`. "
        "video_guided_approach turns the track into grasp and approach candidates."
    ),
    "inputs": [
        {"name": "video", "type": "path", "description": "The human video (Stage 1 input)."},
        {"name": "camera_json", "type": "path", "description": "Stage 2 camera.json (rig frame)."},
        {"name": "time_window_s", "type": "list[2]", "description": "[t0, t1] seconds; for a grasp, from about 1.2 s before the Stage 1 subgoal's start to 1.2 s after it."},
        {"name": "out_dir", "type": "path", "description": "Receives frames/, masks/ and track.json."},
        {"name": "side", "type": "str", "description": "'right' or 'left': the hand to keep when two are visible. Default 'right'."},
        {"name": "prompt", "type": "str", "description": "SAM 3 text prompt. Default 'human hand'."},
        {"name": "stride", "type": "int", "description": "Use every stride-th frame. Default 2."},
        {"name": "plane_z_m, tip_z_m", "type": "float", "description": "Heights of the planes the centroid and the leading point are back-projected onto. Defaults 0.10, 0.08."},
        {"name": "settle_speed_m_s", "type": "float", "description": "Centroid speed below which the hand has settled. Default 0.06."},
        {"name": "min_area_px", "type": "int", "description": "Smaller masks count as no hand. Default 400."},
    ],
    "outputs": [
        {"name": "track", "type": "list", "description": "[{t, centroid [x,y,z], tip [x,y,z], centroid_px, tip_px, area_px}] for frames with a hand."},
        {"name": "entry_s, settle_s", "type": "float | None", "description": "When the hand enters the view and when it settles."},
        {"name": "track_json", "type": "path", "description": "The result as JSON (video_guided_approach's input)."},
    ],
    "status": "tested",
    "entrypoint": "human_hand_track",
    "requirements": ["sam3", "torch", "opencv-python"],
    "tags": ["hand", "video", "track", "approach", "SAM 3"],
    "used_in": ["Pick up the cup and place it on the plate"],
    "cli": "python -m tool_library.skills.human_hand_track VIDEO CAMERA_JSON T0 T1 OUT_DIR",
}

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _camera(camera_json):
    c = json.loads(Path(camera_json).read_text())
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])
    return K, np.asarray(c["R_cv"], float), np.asarray(c["pos"], float), (int(c["width"]), int(c["height"]))


def back_project(K, R, C, u, v, z):
    """World point on the plane at height z seen at pixel (u, v); None when the ray does not reach it."""
    d = R @ np.linalg.solve(K, [float(u), float(v), 1.0])
    if abs(d[2]) < 1e-9:
        return None
    s = (float(z) - C[2]) / d[2]
    return None if s <= 0 else C + s * d


def hand_component(mask, side="right", min_area_px=400):
    """The hand's connected component of a binary mask (None when there is none)."""
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area_px]
    if not big:
        return None
    big.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    if len(big) > 1 and stats[big[1], cv2.CC_STAT_AREA] > 0.4 * stats[big[0], cv2.CC_STAT_AREA]:
        pick = max(big[:2], key=lambda i: cents[i][0]) if side == "right" else min(big[:2], key=lambda i: cents[i][0])
    else:
        pick = big[0]
    return lab == pick


def leading_point(comp):
    """The pixel of the component farthest from where it crosses the image border (the arm's side); without a
    border crossing, the one farthest from its centroid."""
    ys, xs = np.nonzero(comp)
    H, W = comp.shape
    edge = (xs == 0) | (xs == W - 1) | (ys == 0) | (ys == H - 1)
    ref = np.array([xs[edge].mean(), ys[edge].mean()]) if edge.any() else np.array([xs.mean(), ys.mean()])
    d = (xs - ref[0]) ** 2 + (ys - ref[1]) ** 2
    far = d >= np.percentile(d, 97)
    return float(xs[far].mean()), float(ys[far].mean())


def summarize(track, settle_speed_m_s=0.06):
    entry = track[0]["t"] if track else None
    settle = None
    for i in range(1, len(track)):
        rest = track[i:]
        if len(rest) < 2:
            break
        sp = [np.linalg.norm(np.subtract(b["centroid"][:2], a["centroid"][:2])) / max(b["t"] - a["t"], 1e-6)
              for a, b in zip(rest[:-1], rest[1:])]
        if max(sp[:4]) < settle_speed_m_s:
            settle = track[i]["t"]
            break
    return entry, settle


def human_hand_track(video, camera_json, time_window_s, out_dir, side="right", prompt="human hand", stride=2,
                     plane_z_m=0.10, tip_z_m=0.08, settle_speed_m_s=0.06, min_area_px=400, processor=None):
    from tool_library.skills.segment_with_text_prompts import segment_with_text_prompts
    out = Path(out_dir)
    fdir, mdir = out / "frames", out / "masks"
    fdir.mkdir(parents=True, exist_ok=True)
    for old in fdir.glob("frame*.jpg"):
        old.unlink()
    K, R, C, (W, H) = _camera(camera_json)
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    t0, t1 = float(time_window_s[0]), float(time_window_s[1])
    times, i, k = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        if t0 - 1e-9 <= t <= t1 + 1e-9 and (i - int(round(t0 * fps))) % max(1, int(stride)) == 0:
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H))
            cv2.imwrite(str(fdir / f"frame{k:05d}.jpg"), frame)
            times.append(t); k += 1
        i += 1
    cap.release()
    if not times:
        raise ValueError(f"no video frames in {time_window_s}")
    seg = segment_with_text_prompts(str(fdir), [prompt], str(mdir), processor=processor)
    track = []
    for k, t in enumerate(times):
        m = cv2.imread(str(mdir / f"frame{k:05d}.png"), 0)
        comp = hand_component(m > 0, side, min_area_px) if m is not None else None
        if comp is None:
            continue
        ys, xs = np.nonzero(comp)
        cpx = (float(xs.mean()), float(ys.mean()))
        tpx = leading_point(comp)
        pc, pt = back_project(K, R, C, *cpx, plane_z_m), back_project(K, R, C, *tpx, tip_z_m)
        if pc is None or pt is None:
            continue
        track.append({"t": round(t, 4), "centroid": [round(float(v), 4) for v in pc], "tip": [round(float(v), 4) for v in pt],
                      "centroid_px": [round(v, 1) for v in cpx], "tip_px": [round(v, 1) for v in tpx], "area_px": int(comp.sum())})
    entry, settle = summarize(track, settle_speed_m_s)
    res = {"video": str(video), "camera_json": str(camera_json), "time_window_s": [t0, t1], "side": side, "prompt": prompt,
           "planes_z_m": {"centroid": plane_z_m, "tip": tip_z_m}, "n_frames": len(times), "n_tracked": len(track),
           "entry_s": entry, "settle_s": settle, "track": track, "model_identity": seg.get("model_identity"),
           "track_json": str(out / "track.json")}
    (out / "track.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("video"); ap.add_argument("camera_json"); ap.add_argument("t0", type=float); ap.add_argument("t1", type=float)
    ap.add_argument("out_dir"); ap.add_argument("--side", default="right"); ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args(argv)
    r = human_hand_track(a.video, a.camera_json, [a.t0, a.t1], a.out_dir, side=a.side, stride=a.stride)
    print(json.dumps({k: v for k, v in r.items() if k != "track"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
