"""Check the generated G1 + Dex1-1 setup: counts, jaw tracking on the physical runtime, and a top-down cup grasp.

    MUJOCO_GL=egl python setup_g1_dex1/check_setup.py

Prints one [PASS] / [FAIL] line per check and exits non-zero on any failure. Writes images/grasp_cup_<side>.png (the
gripper at the cup grasp, seated) for inspection.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from tool_library.common import episode as E, hand as H, rigid_robot as RR  # noqa: E402

SIDES = ("left", "right")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
CUP_R, PAD_INNER_Y, PAD_CENTRE = 0.035, 0.0229, 0.110
results = []


def check(name, ok, detail):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def counts():
    m = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    rt = RR.compile_rigid(str(HERE / "scene.xml")).model
    check("counts", (m.nq, m.nu, rt.nq, rt.nu) == (30, 16, 18, 18),
          f"proposal nq {m.nq} nu {m.nu}, runtime nq {rt.nq} nu {rt.nu} (expected 30/16, 18/18)")


def jaw_tracking(close_s=2.0, hold_s=1.0, after_s=1.0, fps=30.):
    """Hold home, close both grippers fully in close_s, hold: the jaws, their coupling and the arm joints."""
    P, _ = E.build_execution_model(HERE / "scene.xml")
    R = RR.compile_rigid(P)
    n = int((hold_s + close_s + after_s) * fps) + 1
    q = np.tile(P.key_qpos[0], (n, 1))
    u = np.clip((np.arange(n) / fps - hold_s) / close_s, 0, 1)
    u = u ** 3 * (10 - 15 * u + 6 * u * u)
    closed = {}
    for s in SIDES:
        for f in ("thumb", "index"):
            j = P.joint(f"robot_{s}_{s}_{f}_jaw").id
            a = P.jnt_qposadr[j]
            closed[s] = float(P.jnt_range[j, 1])
            q[:, a] = q[0, a] + u * (closed[s] - q[0, a])
    out = E.servo_rollout(P, q, fps)
    m, ref = R.model, R.to_runtime_qpos(q)
    S, k_end = out["substep_qpos"], int((hold_s + close_s) / m.opt.timestep) - 1
    for s in SIDES:
        th, ix = (m.jnt_qposadr[m.joint(f"robot_{s}_{s}_{f}_jaw").id] for f in ("thumb", "index"))
        ratio = float(S[k_end, th] / closed[s])
        coupling = float(np.abs(S[:, th] - S[:, ix]).max())
        check(f"jaw_tracking_{s}", out["unstable"] is None and ratio >= 0.95 and coupling < 1e-4,
              f"jaws at {S[k_end, th]:.5f} / {closed[s]:.5f} m ({ratio:.3f}) at the end of a {close_s:g} s close, "
              f"{S[-1, th]:.5f} m after {after_s:g} s more; coupling error {coupling * 1e3:.4f} mm")
        k0 = int(0.5 / m.opt.timestep)
        err = {x: float(np.abs(S[k0:, m.jnt_qposadr[m.joint(f"robot_{s}_arm_{x}_joint").id]]
                                - ref[0, m.jnt_qposadr[m.joint(f"robot_{s}_arm_{x}_joint").id]]).max()) for x in ARM}
        check(f"arm_tracking_{s}", err["wrist_roll"] < 0.02 and max(err.values()) < 0.02,
              f"worst joint error after 0.5 s {max(err.values()):.2e} rad (wrist roll {err['wrist_roll']:.2e})")


def top_down_cup(side):
    """The setup's cup, grasped from above at 30 mm below its rim, jaws closing along x, lifted 150 mm."""
    from tool_library.verifiers.grasp_lift_test import grasp_lift_test
    scene = {"hand_xml": str(HERE / f"robot/dex1_{side}.xml"), "side": side, "floor": True,
             "objects": [{"name": "cup", "xml": str(HERE / "assets/test_cup.xml"), "pos": [0, 0, 0.05]}]}
    z, y = np.array([0, 0, -1.]), np.array([1., 0, 0])
    tcp = np.array([0, 0, 0.07])
    model, data0, rig, ids = H.open_scene(scene)
    roles = H.jaw_roles(rig)
    pose = {"pos": (tcp - PAD_CENTRE * z).tolist(),
            "euler": Rotation.from_matrix(np.column_stack([np.cross(y, z), y, z])).as_euler("XYZ").tolist(),
            "finger_targets": {k: PAD_INNER_Y - CUP_R for k in rig.finger_range}, "grip_fingers": list(rig.fingers)}
    r = grasp_lift_test(scene, pose)
    reserve = next((p for p in r["phases"] if p["name"] == "servo_reserve"), {})
    top = max(lf["achieved_mm"] for lf in r["lifts"]) if r["lifts"] else 0.0
    pen = r["penetration"]["max_contact_mm"]
    check(f"cup_top_down_{side}", r["ok"] and top >= 145 and pen < 0.1 and reserve.get("saturated_frac", 1) == 0,
          f"ok {r['ok']}, roles {roles}, lift {top} mm of 150, grip {r['grip']['force_N']} N on {r['grip']['fingers']}, "
          f"servo saturation {reserve.get('saturated_frac')}, peak penetration {pen} mm")
    render_pose(model, data0, rig, pose, HERE / f"images/grasp_cup_{side}.png")


def render_pose(model, data0, rig, pose, path):
    from PIL import Image
    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos
    for k, v in zip(("x", "y", "z"), pose["pos"]):
        d.qpos[rig.wrist_qpos[k]] = v
    for k, v in zip(("rx", "ry", "rz"), pose["euler"]):
        d.qpos[rig.wrist_qpos[k]] = v
    for k, v in pose["finger_targets"].items():
        d.qpos[rig.finger_qpos[k]] = v
    mujoco.mj_forward(model, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.07]
    cam.distance, cam.azimuth, cam.elevation = 0.45, 60.0, -25.0
    with mujoco.Renderer(model, 480, 640) as r:
        r.update_scene(d, camera=cam)
        Image.fromarray(r.render()).save(path)


if __name__ == "__main__":
    counts()
    jaw_tracking()
    for side in SIDES:
        top_down_cup(side)
    print(f"{sum(results)}/{len(results)} checks pass")
    sys.exit(0 if all(results) else 1)
