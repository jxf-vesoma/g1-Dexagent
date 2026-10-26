import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,mujoco
from tool_library.skills.parallel_jaw_grasp_candidates_raygroups import _hit
for case,r,h in [('small',.012,.02),('medium',.025,.04),('large',.035,.05)]:
 xml=f'<mujoco><worldbody><body pos="0 0 .1"><freejoint/><geom name="visual" type="cylinder" size="{r} {h}" group="1" contype="0" conaffinity="0"/><geom name="collision" type="cylinder" size="{r} {h}" group="3"/></body></worldbody></mujoco>'
 m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
 hit=_hit(m,d,np.array([.1,0,.1]),np.array([-1.,0,0]),{1})
 assert hit is not None and abs(hit[0][0]-r)<1e-6,(case,hit)
 print(f'[PASS] {case} group-3 collision hit')