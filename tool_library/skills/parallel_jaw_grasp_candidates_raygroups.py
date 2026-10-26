TOOL={"name":"parallel_jaw_grasp_candidates_raygroups","title":"Generate parallel-jaw grasp candidates with collision-group ray casting","type":"skill","category":"Grasp & Poses","summary":"Find antipodal two-jaw poses on rigid bodies using ray masks derived from the target collision geoms' MuJoCo groups.","description":"Samples jaw axes and in-plane approach directions, ray-casts only target collision geometry groups, validates antipodal normals and jaw width, positions the Dex1 pad midpoint at contact, and screens open approach, retreat, and closure against the scene. This handles assets whose collision geoms use non-default geom groups.","inputs":[{"name":"scene","type":"dict","description":"Hand scene {hand_xml,side,objects,floor}; last object is grasped."},{"name":"n_axes","type":"int","description":"Fibonacci jaw axes."},{"name":"approach_step_deg","type":"float","description":"Approach direction sampling step."},{"name":"depths_m","type":"list","description":"Offsets along approach from target COM."},{"name":"cone_deg","type":"float","description":"Max antipodal normal angle."},{"name":"clearance_m","type":"float","description":"Jaw width spare on each side."},{"name":"retreat_m","type":"float","description":"Open hand retreat distance."},{"name":"max_candidates","type":"int","description":"Maximum candidate poses."}],"outputs":[{"name":"grasps","type":"list","description":"Ranked poses with jaw axis, contacts and width."},{"name":"n_feasible","type":"int","description":"Static feasible candidates."},{"name":"rejected","type":"dict","description":"Counts by rejection reason."}],"status":"tested","entrypoint":"parallel_jaw_grasp_candidates_raygroups","requirements":["mujoco>=3.1","numpy","scipy"],"tags":["parallel-jaw","grasp","raycast","collision-groups"],"used_in":["cup_on_coaster"]}
import warnings,numpy as np
_TOL_MM=.5
def _canon(u):
 u=np.asarray(u,float);u=u/np.linalg.norm(u)
 for k in range(3):
  if abs(u[k])>1e-9:return u if u[k]>0 else -u
 return u
def _fibonacci(n):
 i=np.arange(n)+.5;z=1-2*i/n;r=np.sqrt(np.clip(1-z*z,0,1));p=i*np.pi*(3-np.sqrt(5));return [_canon(u) for u in np.c_[r*np.cos(p),r*np.sin(p),z]]
def _hit(m,d,start,direction,gids):
 import mujoco
 group=np.zeros(6,np.uint8)
 for g in gids: group[int(m.geom_group[g])]=1
 gid,n=np.zeros(1,np.int32),np.zeros(3);p=np.asarray(start,float).copy()
 dist=mujoco.mj_ray(m,d,p,direction,group,1,-1,gid,n)
 if dist<0 or int(gid[0]) not in gids:return None
 return p+dist*direction,n/max(np.linalg.norm(n),1e-12)
def _tcp(m,rig):
 import mujoco
 d=mujoco.MjData(m);ks=list(rig.finger_range)
 def f(q):
  for k in ks:d.qpos[rig.finger_qpos[k]]=q
  mujoco.mj_kinematics(m,d);R=d.xmat[rig.root_body].reshape(3,3)
  return [R.T@(d.site_xpos[s]-d.xpos[rig.root_body]) for s in rig.fingertip_sites.values()]
 a,b=f(0);gap=np.linalg.norm(a-b);a1,b1=f(.01);rate=(gap-np.linalg.norm(a1-b1))/.01
 return (a+b)/2,gap,rate
def parallel_jaw_grasp_candidates_raygroups(scene,n_axes=32,approach_step_deg=30.,depths_m=(0.,-.02,.02),cone_deg=25.,clearance_m=.003,retreat_m=.08,max_candidates=24):
 import mujoco
 from scipy.spatial.transform import Rotation
 from tool_library.common import hand as H
 m,d0,rig,ids=H.open_scene(scene)
 if len(H.jaw_roles(rig))!=2:raise ValueError('scene must use a two-jaw hand')
 jaws=list(rig.finger_range);lo=min(rig.finger_range[k][0] for k in jaws);tcp,gap,rate=_tcp(m,rig);opening=gap-rate*lo
 bid=H.grasp_body(m,scene,ids);gids=set(ids['grasp_gids']);d=mujoco.MjData(m);d.qpos[:]=d0.qpos;mujoco.mj_forward(m,d);com=d.xipos[bid].copy()
 reach=2*max(float(m.geom_rbound[g])+np.linalg.norm(d.geom_xpos[g]-com) for g in gids)+.04
 axes=_fibonacci(int(n_axes))+[_canon(v) for v in np.eye(3)]+[_canon(v) for v in d.xmat[bid].reshape(3,3).T];uniq=[]
 for u in axes:
  if all(abs(u@v)<.999 for v in uniq):uniq.append(u)
 reject={k:0 for k in ('miss','too_wide','not_antipodal','collision','retreat','table_blocks_closure')};out=[]
 def place(pos,eul,q):
  d.qpos[:]=d0.qpos;rig.teleport(d,pos,eul)
  for k in jaws:d.qpos[rig.finger_qpos[k]]=q
  mujoco.mj_forward(m,d)
 for u in uniq:
  e1=np.cross(u,[0,0,1.])
  if np.linalg.norm(e1)<1e-6:e1=np.cross(u,[1.,0,0])
  e1/=np.linalg.norm(e1);e2=np.cross(u,e1)
  for th in np.radians(np.arange(0,360,approach_step_deg)):
   a=np.cos(th)*e1+np.sin(th)*e2
   if a[2]>.05:continue
   R=np.column_stack([np.cross(u,a),u,a])
   with warnings.catch_warnings():
    warnings.simplefilter('ignore',UserWarning);eul=Rotation.from_matrix(R).as_euler('XYZ')
   for s in depths_m:
    c=com+float(s)*a;h1=_hit(m,d0,c+reach*u,-u,gids);h2=_hit(m,d0,c-reach*u,u,gids)
    if h1 is None or h2 is None:reject['miss']+=1;continue
    (p1,n1),(p2,n2)=h1,h2;width=float((p1-p2)@u)
    if width+2*clearance_m>opening:reject['too_wide']+=1;continue
    angle=float(np.degrees(np.arccos(np.clip(min(n1@u,-n2@u),-1,1))))
    if min(n1@u,-n2@u)<np.cos(np.radians(cone_deg)):reject['not_antipodal']+=1;continue
    pos=(p1+p2)/2-R@tcp;q=float(np.clip((gap-width)/rate,lo,max(r[1] for r in rig.finger_range.values())))
    place(pos,eul,lo);cl=H.clearance(m,d,rig,ids)
    if cl['object_mm']>_TOL_MM or cl['floor_mm']>_TOL_MM:reject['collision']+=1;continue
    clear=True
    for k in (1,2,3):
     place(pos-a*retreat_m*k/3,eul,lo);cl=H.clearance(m,d,rig,ids)
     if cl['object_mm']>_TOL_MM or cl['floor_mm']>_TOL_MM:clear=False;break
    if not clear:reject['retreat']+=1;continue
    place(pos,eul,q);cl=H.clearance(m,d,rig,ids,ignore_tips=rig.fingers)
    if cl['floor_mm']>_TOL_MM or cl['object_mm']>_TOL_MM:reject['table_blocks_closure']+=1;continue
    out.append({'side':rig.side,'pos':pos.tolist(),'euler':eul.tolist(),'finger_targets':{k:q for k in jaws},'finger_targets_open':{k:float(lo) for k in jaws},'grip_fingers':list(rig.fingers),'family':'parallel_jaw','jaw_axis':u.tolist(),'approach':(-a).tolist(),'contacts':[p1.tolist(),p2.tolist()],'width_m':width,'antipodal_deg':angle,'depth_m':float(s)})
 out.sort(key=lambda g:(g['antipodal_deg'],-g['approach'][2],abs(g['depth_m'])))
 return {'n_feasible':len(out),'rejected':reject,'opening_m':float(opening),'grasps':out[:int(max_candidates)]}
