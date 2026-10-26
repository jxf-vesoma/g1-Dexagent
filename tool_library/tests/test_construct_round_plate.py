import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco
from tool_library.skills.construct_round_plate import construct_round_plate

def check(tag, **kw):
    with tempfile.TemporaryDirectory() as d:
        r=construct_round_plate(d, **kw)
        m=mujoco.MjModel.from_xml_path(r['xml'])
        assert m.nbody>=2 and abs(r['radius_m']-kw.get('radius',0.04))<1e-12
        assert abs(r['thickness_m']-kw.get('thickness',0.004))<1e-12
        print('[PASS]',tag)
check('free coaster', name='coaster', radius=0.04, thickness=0.003, include_floor=False)
check('fixed plate', name='target', radius=0.09, thickness=0.008, fixed=True, mass=0.2)
check('small puck coloured', name='puck', radius=0.015, thickness=0.012, rgba=[1,0,0,1], yaw_deg=37)
