"""Tests of generate_parallel_jaw_grasps on the Dex1-1 gripper of setup_g1_dex1: an upright cup grasped from above, a
box wider than the jaw opening (no grasp), a cup lying on its side (jaw axis across the cylinder's axis), and a cup
whose visual shell lies on its collision surface (as construct_cup builds it).

    MUJOCO_GL=egl python tool_library/tests/test_generate_parallel_jaw_grasps.py

The gripper MJCF comes from DEXAGENT_JAW_HAND_XML, by default setup_g1_dex1/robot/dex1_right.xml; the test skips
cleanly when it does not exist. Objects are plain MJCF primitives written here.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

HAND = os.environ.get("DEXAGENT_JAW_HAND_XML", str(ROOT / "setup_g1_dex1" / "robot" / "dex1_right.xml"))

CUP = """<mujoco model="cup"><worldbody>
  <body name="cup" pos="0 0 0.05"><freejoint/>
    <geom name="cup_body" type="cylinder" size="0.035 0.05" mass="0.2" rgba="0.85 0.85 0.8 1"/>
  </body></worldbody></mujoco>"""
WIDE_BOX = """<mujoco model="wide_box"><worldbody>
  <body name="wide_box" pos="0 0 0.03"><freejoint/>
    <geom name="wide_box_body" type="box" size="0.06 0.06 0.03" mass="0.3" rgba="0.8 0.45 0.2 1"/>
  </body></worldbody></mujoco>"""
CUP_ON_SIDE = """<mujoco model="lying_cup"><worldbody>
  <body name="lying_cup" pos="0 0 0.035"><freejoint/>
    <geom name="lying_cup_body" type="cylinder" fromto="-0.05 0 0 0.05 0 0" size="0.035" mass="0.2" rgba="0.85 0.85 0.8 1"/>
  </body></worldbody></mujoco>"""
# construct_cup's layout: a visual-only shell (group 1) on exactly the surface of the collision geometry (group 3)
SHELLED_CUP = """<mujoco model="shelled_cup"><worldbody>
  <body name="shelled_cup" pos="0 0 0.05"><freejoint/>
    <geom name="shelled_cup_shell" type="cylinder" size="0.03 0.05" contype="0" conaffinity="0" mass="0" group="1" rgba="0.2 0.3 0.8 1"/>
    <geom name="shelled_cup_body" type="cylinder" size="0.03 0.05" mass="0.08" group="3" rgba="0 0 0 0"/>
  </body></worldbody></mujoco>"""


def scene_for(xml, name, tmp):
    p = Path(tmp) / f"{name}.xml"
    p.write_text(xml)
    return {"hand_xml": HAND, "side": "right", "floor": True, "objects": [{"name": name, "xml": str(p), "pos": [0, 0, 0]}]}


def main():
    if not Path(HAND).exists():
        print(f"SKIP: no parallel-jaw hand at {HAND} (generate setup_g1_dex1/ or set DEXAGENT_JAW_HAND_XML)")
        return 0
    from tool_library.skills.generate_parallel_jaw_grasps import generate_parallel_jaw_grasps
    tmp = tempfile.mkdtemp(prefix="dexagent_jaw_", dir=os.environ.get("TMPDIR"))
    failures = 0
    t0 = time.time()

    def report(name, ok, detail):
        nonlocal failures
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}  ({time.time() - t0:.0f}s)")

    cup = scene_for(CUP, "cup", tmp)
    r = generate_parallel_jaw_grasps(cup, verify=True, verify_top=3)
    top = [g for g in r["grasps"] if g["approach"][2] > 0.9]      # approach: the retreat direction, up
    report("cup_top_down", bool(top),
           f"{r['n_feasible']} feasible of {r['n_candidates']}, {r['n_lift']} of {r['n_verified']} lift, "
           f"{len(top)} lifting from above; best approach {r['grasps'][0]['approach'] if r['grasps'] else None}, "
           f"lift {top[0]['lift']['lifts'][-1] if top else None}")

    box = scene_for(WIDE_BOX, "wide_box", tmp)
    r = generate_parallel_jaw_grasps(box, verify=True, verify_top=3)
    report("box_wider_than_opening", r["n_feasible"] == 0 and not r["grasps"],
           f"{r['n_feasible']} feasible of {r['n_candidates']} (opening {r['opening_m'] * 1000:.1f} mm, box 120 mm); "
           f"rejected {r['rejected']}")

    lying = scene_for(CUP_ON_SIDE, "lying_cup", tmp)
    r = generate_parallel_jaw_grasps(lying, verify=True, verify_top=3)
    axis = np.array([1.0, 0.0, 0.0])
    across = [g for g in r["grasps"] if abs(np.asarray(g["jaw_axis"]) @ axis) < 0.1]
    report("cup_on_side", bool(across),
           f"{r['n_feasible']} feasible, {r['n_lift']} of {r['n_verified']} lift, {len(across)} with the jaw axis across "
           f"the cylinder axis; first {across[0]['jaw_axis'] if across else None}")

    shelled = scene_for(SHELLED_CUP, "shelled_cup", tmp)
    r = generate_parallel_jaw_grasps(shelled, verify=True, verify_top=3)
    report("visual_shell_over_collision", r["n_lift"] > 0,
           f"{r['n_feasible']} feasible of {r['n_candidates']}, {r['n_lift']} of {r['n_verified']} lift; rejected {r['rejected']}")
    print(f"{4 - failures}/4 pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
