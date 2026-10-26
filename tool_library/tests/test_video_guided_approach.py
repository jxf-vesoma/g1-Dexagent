import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from tool_library.skills.video_guided_approach import video_guided_approach, turned_pose
from tool_library.skills import human_hand_track as HT
from tool_library.common import trajectory as TJ

FAILS = []


def check(tag, ok):
    print("[PASS]" if ok else "[FAIL]", tag)
    if not ok:
        FAILS.append(tag)


OBJ = np.array([0.30, 0.20, 0.0])
POSE = {"pos": [0.25, 0.15, 0.15], "euler": [-2.4, 0.5, -1.0], "approach": [-0.5, -0.58, 0.64]}
TRACK = {"entry_s": 1.5, "settle_s": 2.2, "track": [
    {"t": 1.5, "centroid": [0.36, 0.09, 0.1]}, {"t": 1.8, "centroid": [0.33, 0.11, 0.1]},
    {"t": 2.2, "centroid": [0.30, 0.12, 0.1]}, {"t": 2.5, "centroid": [0.30, 0.12, 0.1]}]}

r = video_guided_approach(TRACK, POSE, OBJ, yaw_span_deg=10, yaw_step_deg=5)
best = r["candidates"][0]
az = np.degrees(np.arctan2(*(np.asarray(best["pose"]["pos"][:2]) - OBJ[:2])[::-1]))
check("best candidate puts the hand on the side where the human hand settled", abs(az - (-90.0)) < 1e-6)
check("candidates ranked by closeness to the video", [abs(c["video_offset_deg"]) for c in r["candidates"]] == [0, 5, 5, 10, 10])
inc = np.array(r["video"]["incoming_dir"])
check("incoming direction points back where the hand came from", inc[0] > 0.7 and inc[1] < 0 and abs(np.linalg.norm(inc) - 1) < 1e-3)
p = best["pose"]; ga = np.asarray(p["video_guided"]["gripper_axis"])
b = np.asarray(p["approach"])
check("approach is the unit blend of incoming and gripper axis", abs(np.linalg.norm(b) - 1) < 1e-9 and b @ ga > 0.3 and b @ inc > 0.3)
check("final run along the gripper axis", np.allclose(p["approach_waypoints"][0], 0.05 * ga, atol=1e-4))
check("reach time from the track", abs(r["video"]["reach_s"] - 0.7) < 1e-9)
q = turned_pose(POSE, OBJ[:2], 37.0)
R0, R1 = TJ.euler_to_mat(np.asarray(POSE["euler"])), TJ.euler_to_mat(np.asarray(q["euler"]))
c, s = np.cos(np.radians(37)), np.sin(np.radians(37)); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
check("turned pose: orientation and position turned about the object axis",
      np.allclose(R1, Rz @ R0, atol=1e-9) and abs(np.linalg.norm(np.subtract(q["pos"][:2], OBJ[:2])) - np.linalg.norm(np.subtract(POSE["pos"][:2], OBJ[:2]))) < 1e-9
      and abs(q["pos"][2] - POSE["pos"][2]) < 1e-12)
try:
    video_guided_approach({"track": [TRACK["track"][0], TRACK["track"][0]]}, POSE, OBJ); check("still hand rejected", False)
except ValueError:
    check("still hand rejected", True)

# the tracker on a synthetic video: a hand entering from the right edge and stopping; a stand-in segmenter
cam = {"width": 160, "height": 120, "fx": 190.0, "fy": 190.0, "cx": 80.0, "cy": 60.0, "pos": [0.2, 0.05, 0.5],
       "R_cv": [[1, 0, 0], [0, -0.9453478, 0.3260637], [0, -0.3260637, -0.9453478]]}


class FakeSeg:
    model_identity = {"version": "fake"}

    def set_image(self, img):
        a = np.asarray(img)
        return {"m": (a[:, :, 0] > 128)}

    def set_text_prompt(self, prompt, state):
        import types
        m = state["m"][None, None]
        return {"masks": m, "scores": np.ones(1)}


with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    (d / "camera.json").write_text(json.dumps(cam))
    vw = cv2.VideoWriter(str(d / "v.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 120))
    for i in range(60):
        f = np.zeros((120, 160, 3), np.uint8)
        x0 = max(100, 160 - 4 * i) if i >= 10 else 999
        if x0 < 160:
            f[70:90, x0:160] = 255
        vw.write(f)
    vw.release()
    t = HT.human_hand_track(d / "v.avi", d / "camera.json", [0.0, 1.9], d / "out", stride=1, processor=FakeSeg())
    check("tracker: hand enters when it appears", t["entry_s"] is not None and abs(t["entry_s"] - 11 / 30) < 0.05)
    check("tracker: hand settles once it stops", t["settle_s"] is not None and t["settle_s"] < 1.0)
    xs = [p["centroid"][0] for p in t["track"]]
    check("tracker: centroid moves leftward (from the right edge) on the table", xs[0] > xs[-1])
    check("tracker: leading point ahead of the centroid", all(p["tip_px"][0] < p["centroid_px"][0] for p in t["track"]))
    pt = HT.back_project(np.array([[190.0, 0, 80], [0, 190.0, 60], [0, 0, 1]]), np.asarray(cam["R_cv"]), np.asarray(cam["pos"]), 80, 60, 0.0)
    check("back-projection lands on the requested plane", pt is not None and abs(pt[2]) < 1e-9)

print(f"{len(FAILS)} FAILED" if FAILS else "all passed")
sys.exit(1 if FAILS else 0)
