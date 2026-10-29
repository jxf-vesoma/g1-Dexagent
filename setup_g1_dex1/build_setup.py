"""Generate setup_g1_dex1/: the Unitree G1's two arms with a Unitree Dex1-1 parallel-jaw gripper on each wrist, as a
DexAgent setup beside the default bench in setup/.

    python setup_g1_dex1/build_setup.py --models /path/to/vesoma-models

Reads robots/g1_with_dex1/g1_with_dex1.mujoco.xml of a vesoma-models checkout (its meshes must be real files, not Git
LFS pointers) and writes, next to this file:

    scene.xml               the proposal scene: table, the G1 torso (visual), both arms, both grippers on virtual wrists
    setup.yaml              the setup, machine-readable (the same keys as setup/setup.yaml)
    robot/dex1_<side>.xml   one gripper alone on the six-joint virtual wrist, for the grasp tools
    robot/g1_arm_<side>.xml one arm alone (shoulder to wrist flange), whose inertias the runtime compiler restores
    assets/g1, assets/dex1  the meshes, copied unchanged, and each multi-piece visual mesh split into its connected
                            pieces (<mesh>_partNN.stl; sha256 in assets/LICENSES.md)
    assets/test_cup.xml     the cup the setup's grasp check lifts
    images/*.png            setup/render_setup.py's views at the home keyframe

Every number comes from the source model except the layout below (where the G1 stands at the table, the home pose)
and the arm servo parameters, which are DexAgent's (see ARM below).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SIDES = ("left", "right")
SOURCE = "robots/g1_with_dex1/g1_with_dex1.mujoco.xml"
PARTS_DIR = Path(tempfile.mkdtemp(prefix="g1_dex1_mesh_parts_"))     # staging for the split visual meshes

# The G1 arm, shoulder to wrist flange: source link / joint stems, in chain order. The last link carries the
# wrist yaw joint and the gripper; it is the station's flange (robot_<side>_arm_gripper), as joint 6 of the YAM.
ARM_LINKS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
# DexAgent's motor model of the arm. Torque caps are the G1's actuator limits (the source's actuatorfrcrange: 25 N m
# for the shoulder, elbow and wrist roll motors, 5 N m for wrist pitch and yaw). Posture gains drive the arm's
# one-dimensional nullspace (the elbow swivel) toward the planned joints; damping is the YAM runtime's 0.1.
ARM = {"kp": [60., 60., 40., 40., 20., 8., 8.], "kd": [6., 6., 4., 4., 2., .8, .8], "damping": [.1] * 7}

# Layout, in the rig frame of setup.yaml (x toward the right arm, y into the work, z up, tabletop at z = 0).
SHOULDER_Z = 0.30           # shoulder pitch axes 30 cm above the tabletop: a G1 (1.32 m) at a 0.78 m table
TABLE_NEAR_Y = 0.10         # the table's near edge, 10 cm in front of the shoulders; the torso stays clear of it
TABLE_HALF = (0.6, 0.40, 0.02)    # the lab table: 1.2 m wide, 0.8 m deep
# The real head camera (D435i colour, 640 x 480, fx = fy = 612.48 px, centred, no distortion) against the model's:
# fitted to a real frame of the 10 cm coaster, the 6 x 10 cm cup and the table's far edge
# (runs/g1_cup_on_coaster/real_head/), it sits 1.1 cm lower and looks 2.8 deg less steeply down (a slight lean of the
# standing robot, or the mount's tolerance). Applied in the camera frame: height, then turns about its x and z axes.
HEAD_CAM_FY_PX = 612.48
HEAD_CAM_DZ = -0.0106
HEAD_CAM_TURN_DEG = (2.77, 0.25)
# Home: each gripper resting low beside the work, as the demonstrator's hand rests on the table: 23 cm out to its
# side, 6 cm in front of the shoulders and 8 cm above the table (the casting's underside ~4 cm up), its jaws pointing
# into the work 10 deg down, jaws half open. The jaw tips end ~20 cm in front, clear of a 6 cm cup up to 13 cm from the middle.
HOME_HAND = {"left": (-0.23, 0.06, 0.08), "right": (0.23, 0.06, 0.08)}
HOME_PITCH_DEG = 10.0
HOME_TORSO_CLEARANCE = 0.05     # of the arms that hold the home pose, those keeping the arm 5 cm off the torso first
HOME_SEED = {"left": [-0.6, 0.25, 0.0, 0.9, 0.0, 0.0, 0.0], "right": [-0.6, -0.25, 0.0, 0.9, 0.0, 0.0, 0.0]}
HOME_RESTARTS = 200

# The gripper inside its hand frame. DexAgent's hand frame has +z along the approach (the hand retreats along -z);
# the Dex1-1 casting has x along its fingers, so the casting sits in the hand frame turned -90 deg about y.
DEX1_IN_HAND = Rotation.from_euler("y", -90, degrees=True)
# Jaws: finger 1 (-y side of the casting) is the thumb, finger 2 the index. The source joints open toward the high end
# of their range; DexAgent's fingers close toward it, so each jaw's axis is reversed and its range negated.
JAWS = (("thumb", "finger_link_1", "finger_joint_1", "pad_1", -1.0), ("index", "finger_link_2", "finger_joint_2", "pad_2", 1.0))
JAW_KP, JAW_KV, JAW_FORCE_N = 2000.0, 100.0, 20.0     # the source's Dex1-1 position servo
PAD_CENTRE_X = 0.110                                     # the source's TCP: the middle of the pads, along the fingers
PAD_INNER_Y = 0.0229                                     # the pads' inner faces at jaw 0 (measured on dex1_col_*.stl)

TEST_CUP = """<mujoco model="test_cup">
  <!-- The setup's grasp check: a 70 mm cup, 100 mm tall, 0.2 kg (fits in the Dex1-1's 89 mm opening). -->
  <worldbody>
    <body name="cup" pos="0 0 0.05">
      <freejoint/>
      <geom name="cup_body" type="cylinder" size="0.035 0.05" mass="0.2" rgba="0.85 0.85 0.8 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def f(v, digits=9):
    return " ".join(format(float(x), f".{digits}g") for x in np.atleast_1d(np.asarray(v, float)).ravel())


def quat_of(R):
    q = Rotation.from_matrix(np.asarray(R, float)).as_quat()        # x y z w
    q = np.array([q[3], q[0], q[1], q[2]])
    return q if q[0] >= 0 else -q


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Source:
    """The vesoma-models G1 + Dex1-1 model: its spec (raw mesh-geom frames), compiled model (resolved defaults, inertias)
    and data at qpos0."""

    def __init__(self, models: Path):
        self.path = (models / SOURCE).resolve()
        if not self.path.is_file():
            raise SystemExit(f"{self.path} not found: pass --models <vesoma-models checkout>")
        # absolute path: MuJoCo resolves the nested <include>s of a relative path against the wrong folder
        self.spec = mujoco.MjSpec.from_file(str(self.path))
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.meshes = {}
        base = Path(self.spec.modelfiledir)
        for me in self.spec.meshes:
            p = Path(me.file)
            p = p if p.is_absolute() else base / (self.spec.meshdir or "") / p
            head = p.read_bytes()[:64]
            if head.startswith(b"version https://git-lfs"):
                raise SystemExit(f"{p} is a Git LFS pointer: run `git lfs pull` in the models checkout")
            self.meshes[me.name] = p
        self.spec_meshes = set(self.meshes)
        self._parts = {}

    def parts(self, mesh, out_dir: Path):
        """Names of the connected pieces of a visual mesh, each written to out_dir as its own STL in the source mesh's
        frame (registered in self.meshes), or [mesh] when it is one piece. Collision audits take each robot geom's
        convex hull, and the hull of a several-piece mesh (a finger with its full-width rack, the torso with its
        shoulder housings) fills the gaps between the pieces."""
        if mesh not in self._parts:
            import trimesh
            pieces = trimesh.load_mesh(str(self.meshes[mesh])).split(only_watertight=False)
            # stray triangles and flat slivers (MuJoCo cannot hull them) are left out
            pieces = [p for p in pieces if len(p.vertices) >= 4 and
                      np.linalg.svd(p.vertices - p.vertices.mean(0), compute_uv=False)[-1] > 1e-5] if len(pieces) > 1 else pieces
            if len(pieces) <= 1:
                self._parts[mesh] = [mesh]
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                names = []
                for k, piece in enumerate(pieces):
                    name = f"{mesh}_part{k:02d}"
                    path = out_dir / f"{self.meshes[mesh].stem}_part{k:02d}.stl"
                    path.write_bytes(piece.export(file_type="stl"))
                    self.meshes[name] = path
                    names.append(name)
                self._parts[mesh] = names
        return self._parts[mesh]

    def body(self, name):
        return self.model.body(name).id

    def inertial(self, b):
        m = self.model
        return (f'<inertial pos="{f(m.body_ipos[b])}" quat="{f(m.body_iquat[b])}" mass="{f(m.body_mass[b])}" '
                f'diaginertia="{f(m.body_inertia[b])}"/>')

    def visual_geoms(self, b, mesh_name, rgba_override=None, group=1):
        """The body's visual geoms (group <= 2) as MJCF: meshes with the spec's raw frame (the compiled one includes
        the mesh's recentring), primitives with the compiled frame."""
        m, out = self.model, []
        sb = self.spec.body(m.body(b).name)
        for k, sg in enumerate(sb.geoms):
            g = int(m.body_geomadr[b]) + k
            if int(m.geom_group[g]) > 2:
                continue
            rgba = rgba_override if rgba_override is not None else (
                m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g])
            common = f'contype="0" conaffinity="0" group="{group}" density="0" rgba="{f(rgba, 4)}"'
            if sg.type == mujoco.mjtGeom.mjGEOM_MESH:
                for piece in self.parts(sg.meshname, PARTS_DIR):
                    out.append(f'<geom type="mesh" mesh="{mesh_name(piece)}" pos="{f(sg.pos)}" quat="{f(sg.quat)}" {common}/>')
            else:
                t = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
                     mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule", mujoco.mjtGeom.mjGEOM_SPHERE: "sphere"}[sg.type]
                out.append(f'<geom type="{t}" size="{f(m.geom_size[g][:{"box": 3, "sphere": 1}.get(t, 2)])}" '
                           f'pos="{f(m.geom_pos[g])}" quat="{f(m.geom_quat[g])}" {common}/>')
        return out

    def joint(self, j, name, *, damping=None, extra=""):
        m = self.model
        dof = int(m.jnt_dofadr[j])
        lo, hi = m.jnt_range[j]
        dmp = float(m.dof_damping[dof]) if damping is None else damping
        return (f'<joint name="{name}" axis="{f(m.jnt_axis[j])}" range="{f([lo, hi])}" armature="{f(m.dof_armature[dof])}" '
                f'frictionloss="{f(m.dof_frictionloss[dof])}" damping="{f(dmp)}" '
                f'actuatorfrcrange="{f(m.jnt_actfrcrange[j])}"{extra}/>')


# ----------------------------------------------------------------------------------------------------------- the arm
def arm_tree(src: Source, side: str, name, mesh_name, proposal: bool) -> str:
    """The arm from the shoulder pitch link to the wrist flange, as nested MJCF bodies."""
    m = src.model
    open_tags, close = [], []
    for i, stem in enumerate(ARM_LINKS):
        b = src.body(f"{side}_{stem}_link")
        j = m.joint(f"{side}_{stem}_joint").id
        bname = name("gripper" if stem == "wrist_yaw" else f"{stem}_link")
        lines = [f'<body name="{bname}" pos="{f(m.body_pos[b])}" quat="{f(m.body_quat[b])}">', src.inertial(b),
                 src.joint(j, name(f"{stem}_joint"), damping=5.0 if proposal else 0.1)]
        lines += src.visual_geoms(b, mesh_name)
        open_tags.append("\n".join(lines))
        close.append("</body>")
    return "\n".join(open_tags) + "\n" + "\n".join(reversed(close))


# ------------------------------------------------------------------------------------------------------- the gripper
def mount_in_flange(src: Source, side: str):
    """(p, R) of the hand frame in the flange (wrist yaw link) frame: the casting's offset, turned into the hand frame."""
    b = src.body(f"{side}_dex1_base_link")
    p = np.asarray(src.model.body_pos[b], float)
    R_cast = Rotation.from_quat(np.r_[src.model.body_quat[b][1:], src.model.body_quat[b][0]])
    return p, (R_cast * DEX1_IN_HAND.inv()).as_matrix()


def wrist_camera(src: Source, side: str):
    """(pos, quat, fovy) of the Dex1-1 wrist camera in the hand frame."""
    m, d = src.model, src.data
    c = m.camera(f"camera/wrist_{side}/color").id
    b = src.body(f"{side}_dex1_base_link")
    Rb = d.xmat[b].reshape(3, 3)
    p_cast = Rb.T @ (d.cam_xpos[c] - d.xpos[b])
    R_cast = Rb.T @ d.cam_xmat[c].reshape(3, 3)
    return DEX1_IN_HAND.apply(p_cast), quat_of(DEX1_IN_HAND.as_matrix() @ R_cast), float(m.cam_fovy[c])


def hand_tree(src: Source, side: str, prefix: str, mesh_name) -> str:
    """Everything the gripper's hand frame carries, as MJCF children of <side>_base_yaw. `prefix` is '' for the
    standalone hand file and 'robot_<side>_' in the scene (MuJoCo's attach naming: prefix + the hand file's name)."""
    m = src.model
    h = f"{prefix}{side}_"
    base = src.body(f"{side}_dex1_base_link")
    cam_body = src.body(f"{side}_dex1_wrist_cam")
    cp, cq, cfov = wrist_camera(src, side)
    out = [f'<camera name="wrist_{side}" pos="{f(cp)}" quat="{f(cq)}" fovy="{f(cfov)}"/>',
           f'<body name="{h}palm" quat="{f(quat_of(DEX1_IN_HAND.as_matrix()))}">', src.inertial(base)]
    out += src.visual_geoms(base, mesh_name)
    sg = src.spec.geom(f"{side}_dex1_base_collision")
    out.append(f'<geom name="{prefix}collision_hand_{side}_palm_0" type="mesh" mesh="{mesh_name(sg.meshname)}" '
               f'pos="{f(sg.pos)}" quat="{f(sg.quat)}" contype="0" conaffinity="0" condim="1" group="3" density="0" '
               f'rgba="0 1 0 1"/>')
    out.append(f'<site name="{h}tcp" pos="{f([PAD_CENTRE_X, 0, 0])}" size="0.006" group="3" rgba="1 0 0 1"/>')
    # the wrist camera's housing and lens barrel: visual, plus the housing as part of the palm's collision
    out.append(f'<body name="{h}wrist_cam" pos="{f(m.body_pos[cam_body])}" quat="{f(m.body_quat[cam_body])}">')
    out.append(src.inertial(cam_body))
    out += src.visual_geoms(cam_body, mesh_name)
    for k, part in enumerate(("housing", "barrel"), start=1):
        name = f"{side}_dex1_wrist_cam_{part}_collision"
        g, sg = m.geom(name).id, src.spec.geom(name)
        common = f'contype="0" conaffinity="0" condim="1" group="3" density="0" rgba="0 1 0 1"'
        if sg.type == mujoco.mjtGeom.mjGEOM_MESH:
            out.append(f'<geom name="{prefix}collision_hand_{side}_palm_{k}" type="mesh" mesh="{mesh_name(sg.meshname)}" '
                       f'pos="{f(sg.pos)}" quat="{f(sg.quat)}" {common}/>')
        else:
            t = {mujoco.mjtGeom.mjGEOM_BOX: ("box", 3), mujoco.mjtGeom.mjGEOM_CYLINDER: ("cylinder", 2)}[sg.type]
            out.append(f'<geom name="{prefix}collision_hand_{side}_palm_{k}" type="{t[0]}" size="{f(m.geom_size[g][:t[1]])}" '
                       f'pos="{f(m.geom_pos[g])}" quat="{f(m.geom_quat[g])}" {common}/>')
    out.append("</body>")
    for finger, link, joint, pad, y_sign in JAWS:
        b = src.body(f"{side}_dex1_{link}")
        j = m.joint(f"{side}_dex1_{joint}").id
        dof = int(m.jnt_dofadr[j])
        lo, hi = m.jnt_range[j]
        out.append(f'<body name="{h}{finger}_jaw_link">')
        out.append(src.inertial(b))
        out.append(f'<joint name="{h}{finger}_jaw" type="slide" axis="{f(-m.jnt_axis[j])}" range="{f([-hi, -lo])}" '
                   f'armature="{f(m.dof_armature[dof])}" damping="{f(m.dof_damping[dof])}" '
                   f'frictionloss="{f(m.dof_frictionloss[dof])}" actuatorfrcrange="{f(m.jnt_actfrcrange[j])}"/>')
        out += src.visual_geoms(b, mesh_name)
        pg = src.spec.geom(f"{side}_dex1_{pad}")
        out.append(f'<geom name="{prefix}collision_hand_{side}_{finger}_0" type="mesh" mesh="{mesh_name(pg.meshname)}" '
                   f'pos="{f(pg.pos)}" quat="{f(pg.quat)}" contype="0" conaffinity="0" condim="1" group="3" density="0" '
                   f'rgba="0 1 0 1"/>')
        tip = [PAD_CENTRE_X, y_sign * PAD_INNER_Y, 0.0]
        out.append(f'<site name="{h}{finger}_tip" pos="{f(tip)}" size="0.005" group="3" rgba="1 0 0 1"/>')
        out.append("</body>")
    out.append("</body>")
    return "\n".join(out)


def wrist_chain(side: str, prefix: str, inner: str) -> str:
    """The six virtual wrist joints (three slides, three hinges in intrinsic XYZ order) around the hand."""
    h = f"{prefix}{side}_"
    links = (("base_tx", "pos_x", "slide", "1 0 0", "-5 5"), ("base_ty", "pos_y", "slide", "0 1 0", "-5 5"),
             ("base_tz", "pos_z", "slide", "0 0 1", "-5 5"), ("base_roll", "rot_x", "hinge", "1 0 0", "-6.28 6.28"),
             ("base_pitch", "rot_y", "hinge", "0 1 0", "-6.28 6.28"), ("base_yaw", "rot_z", "hinge", "0 0 1", "-6.28 6.28"))
    out = []
    for body, joint, typ, axis, rng in links:
        out.append(f'<body name="{h}{body}">\n<inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01"/>\n'
                   f'<joint name="{h}{joint}" type="{typ}" axis="{axis}" range="{rng}" armature="1"/>')
    return "\n".join(out) + "\n" + inner + "\n" + "</body>\n" * len(links)


def hand_actuators(src: Source, side: str, prefix: str) -> str:
    h = f"{prefix}{side}_"
    out = []
    for joint, rng in (("pos_x", "-5 5"), ("pos_y", "-5 5"), ("pos_z", "-5 5"),
                       ("rot_x", "-6.28 6.28"), ("rot_y", "-6.28 6.28"), ("rot_z", "-6.28 6.28")):
        out.append(f'<general name="{h}{joint}_position" joint="{h}{joint}" ctrllimited="true" ctrlrange="{rng}" '
                   f'biastype="affine" gainprm="1000" biasprm="0 -1000 0"/>')
    for finger, _, joint, _, _ in JAWS:
        lo, hi = src.model.jnt_range[src.model.joint(f"{side}_dex1_{joint}").id]
        out.append(f'<general name="{h}{finger}_jaw_position" joint="{h}{finger}_jaw" ctrllimited="true" '
                   f'ctrlrange="{f([-hi, -lo])}" forcelimited="true" forcerange="{f([-JAW_FORCE_N, JAW_FORCE_N])}" '
                   f'biastype="affine" gainprm="{f(JAW_KP)}" biasprm="0 {f(-JAW_KP)} {f(-JAW_KV)}"/>')
    return "\n".join(out)


def jaw_coupling(side: str, prefix: str) -> str:
    # one motor drives both jaws on the real gripper: the index jaw follows the thumb jaw (equal values close centred)
    h = f"{prefix}{side}_"
    return (f'<joint name="{h}jaw_coupling" joint1="{h}index_jaw" joint2="{h}thumb_jaw" polycoef="0 1 0 0 0" '
            f'solref="0.005 1"/>')


def mesh_assets(src: Source, names, folder: str) -> str:
    # a split-off piece can be a flat plate or a few loose triangles: its (unused, density 0) inertia as a shell
    shell = ' inertia="shell"'
    return "\n".join(f'<mesh name="{n}" file="{folder}/{src.meshes[n].name}"{"" if n in src.spec_meshes else shell}/>'
                     for n in names)


# ----------------------------------------------------------------------------------------------------------- layout
def torso_pose(src: Source):
    """World (rig) pose of the G1 torso link: turned to face +y, shoulders SHOULDER_Z above the table at y = 0."""
    m = src.model
    sh = np.asarray(m.body_pos[src.body("left_shoulder_pitch_link")], float)   # in the torso frame
    R = Rotation.from_euler("z", 90, degrees=True)
    p = -R.apply([sh[0], 0.0, sh[2]]) + np.array([0.0, 0.0, SHOULDER_Z])
    return p, R


def hand_home_R():
    """Home hand orientation: approach (+z) into the work and HOME_PITCH_DEG down, jaws closing along world x, the
    wrist camera on top."""
    a = np.radians(HOME_PITCH_DEG)
    z = np.array([0.0, np.cos(a), -np.sin(a)])
    y = np.array([-1.0, 0.0, 0.0])
    return np.column_stack([np.cross(y, z), y, z])


def solve_home(scene: Path, mounts):
    """Arm joints placing each hand at its home pose, and the full home qpos with the virtual wrists at the same hand
    poses. Damped least squares within the joint limits from HOME_SEED and HOME_RESTARTS random seeds; of the arms
    that hold the pose, those whose upper arm, forearm and wrist stay HOME_TORSO_CLEARANCE off the torso (the elbow
    swivel otherwise often lays the upper arm against the chest), and of those the one farthest from its joint stops
    (the executor's IK follows the branch it starts on, so a home near a stop runs into it on the first reach)."""
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    q = m.qpos0.copy()
    Rt = hand_home_R()
    rng = np.random.default_rng(0)
    body = lambda g: m.body(int(m.geom_bodyid[g])).name
    torso = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and body(g).startswith("robot_")
             and not body(g).startswith(tuple(f"robot_{s}" for s in SIDES))]
    fromto = np.zeros(6)
    for side in SIDES:
        arm = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
               and body(g).startswith(f"robot_{side}_arm_") and not body(g).endswith(("shoulder_pitch_link", "shoulder_roll_link"))]
        joints = [m.joint(f"robot_{side}_arm_{s}_joint").id for s in ARM_LINKS]
        qa = [int(m.jnt_qposadr[j]) for j in joints]
        da = [int(m.jnt_dofadr[j]) for j in joints]
        lo, hi = m.jnt_range[joints, 0], m.jnt_range[joints, 1]
        fl = m.body(f"robot_{side}_arm_gripper").id
        pm, Rm = mounts[side]
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

        def hand_error(x):
            d.qpos[:] = q; d.qpos[qa] = x; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            Rf = d.xmat[fl].reshape(3, 3)
            ph, Rh = d.xpos[fl] + Rf @ pm, Rf @ Rm
            return ph, np.asarray(HOME_HAND[side]) - ph, Rotation.from_matrix(Rt @ Rh.T).as_rotvec()

        def descend(seed):
            x = np.asarray(seed, float).copy()
            for _ in range(400):
                ph, ep, er = hand_error(x)
                if np.linalg.norm(ep) < 1e-7 and np.linalg.norm(er) < 1e-7:
                    break
                mujoco.mj_jac(m, d, jp, jr, ph, fl)
                J = np.vstack([jp[:, da], jr[:, da]])
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), np.r_[ep, er])
                dq += (np.eye(len(da)) - np.linalg.pinv(J) @ J) @ (0.05 * (np.asarray(seed) - x))
                x = np.clip(x + np.clip(dq, -0.1, 0.1), lo + 1e-3, hi - 1e-3)
            _, ep, er = hand_error(x)
            return x, np.linalg.norm(ep) <= 1e-5 and np.degrees(np.linalg.norm(er)) <= 1e-3

        seeds = [HOME_SEED[side]] + [rng.uniform(lo, hi) for _ in range(HOME_RESTARTS)]
        def torso_clearance(x):
            d.qpos[:] = q; d.qpos[qa] = x; mujoco.mj_kinematics(m, d)
            return min(mujoco.mj_geomDistance(m, d, a, t, HOME_TORSO_CLEARANCE, fromto) for a in arm for t in torso)

        held = [x for x, ok in map(descend, seeds) if ok]
        q[qa] = max(held, key=lambda x: (min(torso_clearance(x), HOME_TORSO_CLEARANCE),
                                         float(np.minimum(x - lo, hi - x).min()))) if held else descend(HOME_SEED[side])[0]
        if held and torso_clearance(q[qa]) < HOME_TORSO_CLEARANCE:
            print(f"warning: the {side} arm's home comes {torso_clearance(q[qa]) * 1000:.0f} mm from the torso")
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        Rf = d.xmat[fl].reshape(3, 3)
        ph, Rh = d.xpos[fl] + Rf @ pm, Rf @ Rm
        res_p = float(np.linalg.norm(np.asarray(HOME_HAND[side]) - ph))
        res_r = float(np.degrees(Rotation.from_matrix(Rt @ Rh.T).magnitude()))
        if res_p > 1e-5 or res_r > 1e-3:
            raise SystemExit(f"home pose of the {side} hand is not reachable: {res_p * 1000:.2f} mm, {res_r:.3f} deg")
        wrist = np.r_[ph, Rotation.from_matrix(Rh).as_euler("XYZ")]
        for k, jn in enumerate(("pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z")):
            q[int(m.jnt_qposadr[m.joint(f"robot_{side}_{side}_{jn}").id])] = wrist[k]
    ctrl = np.zeros(m.nu)
    for a in range(m.nu):
        ctrl[a] = q[int(m.jnt_qposadr[int(m.actuator_trnid[a, 0])])]
    return q, ctrl, m


REACH_TCP_Z = 0.05          # the work area: where the pad centre reaches 5 cm above the table, gripper pointing down
REACH_STEP = 0.05


def reach_grid(m, home_q, mounts):
    """Per side, the table points (x, y) where the arm holds the gripper vertical, jaws along x, its pad centre
    REACH_TCP_Z above the table, within the joint limits (damped least squares from home)."""
    d = mujoco.MjData(m)
    z, y = np.array([0, 0, -1.]), np.array([1., 0, 0])
    Rt = np.column_stack([np.cross(y, z), y, z])
    xs = np.round(np.arange(-0.40, 0.40 + 1e-9, REACH_STEP), 3)
    ys = np.round(np.arange(TABLE_NEAR_Y, TABLE_NEAR_Y + 2 * TABLE_HALF[1] + 1e-9, REACH_STEP), 3)
    out = {}
    for side in SIDES:
        js = [m.joint(f"robot_{side}_arm_{s}_joint").id for s in ARM_LINKS]
        qa, da = m.jnt_qposadr[js], m.jnt_dofadr[js]
        fl = m.body(f"robot_{side}_arm_gripper").id
        pm, Rm = mounts[side]
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        pts = []
        for x in xs:
            for yy in ys:
                target = np.array([x, yy, REACH_TCP_Z]) - PAD_CENTRE_X * Rt[:, 2]
                q = home_q.copy()
                for _ in range(300):
                    d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
                    Rf = d.xmat[fl].reshape(3, 3)
                    ph, Rh = d.xpos[fl] + Rf @ pm, Rf @ Rm
                    e = np.r_[target - ph, Rotation.from_matrix(Rt @ Rh.T).as_rotvec()]
                    if np.linalg.norm(e) < 1e-5:
                        pts.append((float(x), float(yy)))
                        break
                    mujoco.mj_jac(m, d, jp, jr, ph, fl)
                    J = np.vstack([jp[:, da], jr[:, da]])
                    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
                    q[qa] = np.clip(q[qa] + np.clip(dq, -.2, .2), m.jnt_range[js, 0], m.jnt_range[js, 1])
        out[side] = pts
    return out


# ------------------------------------------------------------------------------------------------------------ files
def write_scene(src: Source, home=None):
    m = src.model
    tp, tR = torso_pose(src)
    torso = src.body("torso_link")
    g1_meshes, dex_meshes = [], []

    def g1_mesh(n):
        if n not in g1_meshes:
            g1_meshes.append(n)
        return f"g1_{n}"

    def dex_mesh(n):
        if n not in dex_meshes:
            dex_meshes.append(n)
        return n

    bodies = [f'<body name="robot_torso" pos="{f(tp)}" quat="{f(quat_of(tR.as_matrix()))}">']
    bodies += src.visual_geoms(torso, g1_mesh)
    bodies.append("</body>")
    mounts = {s: mount_in_flange(src, s) for s in SIDES}
    for side in SIDES:
        bodies.append(f'<body name="robot_{side}_arm_root" pos="{f(tp)}" quat="{f(quat_of(tR.as_matrix()))}">')
        bodies.append(arm_tree(src, side, lambda n, s=side: f"robot_{s}_arm_{n}", g1_mesh, proposal=True))
        bodies.append("</body>")
    for side in SIDES:
        bodies.append(wrist_chain(side, f"robot_{side}_", hand_tree(src, side, f"robot_{side}_", dex_mesh)))
    hand_geoms = [f"robot_{s}_collision_hand_{s}_{k}" for s in SIDES
                  for k in ("palm_0", "palm_1", "palm_2", "thumb_0", "index_0")]
    pairs = "\n".join(f'<pair name="{g}_g_table" geom1="g_table" geom2="{g}" condim="3" solref="0.01" friction="1 1 0.1 0 0"/>'
                      for g in hand_geoms)
    welds = "\n".join(f'<weld body1="robot_{s}_arm_gripper" body2="robot_{s}_{s}_base_yaw" '
                      f'relpose="{f(mounts[s][0])} {f(quat_of(mounts[s][1]))}" torquescale="0"/>' for s in SIDES)
    couplings = "\n".join(jaw_coupling(s, f"robot_{s}_") for s in SIDES)
    actuators = "\n".join(hand_actuators(src, s, f"robot_{s}_") for s in SIDES)
    arm_custom = []
    for side in SIDES:
        joints = [m.joint(f"{side}_{s}_joint").id for s in ARM_LINKS]
        torque = [float(m.jnt_actfrcrange[j, 1]) for j in joints]
        armature = [float(m.dof_armature[m.jnt_dofadr[j]]) for j in joints]
        arm_custom += [
            f'<text name="dexagent_arm_joints_{side}" data="{" ".join(f"robot_{side}_arm_{s}_joint" for s in ARM_LINKS)}"/>',
            f'<text name="dexagent_arm_xml_{side}" data="robot/g1_arm_{side}.xml"/>',
            f'<numeric name="dexagent_arm_torque_nm_{side}" data="{f(torque)}"/>',
            f'<numeric name="dexagent_arm_kp_{side}" data="{f(ARM["kp"])}"/>',
            f'<numeric name="dexagent_arm_kd_{side}" data="{f(ARM["kd"])}"/>',
            f'<numeric name="dexagent_arm_armature_{side}" data="{f(armature)}"/>',
            f'<numeric name="dexagent_arm_damping_{side}" data="{f(ARM["damping"])}"/>']
    ego = ego_camera(src)
    table_c = [0.0, TABLE_NEAR_Y + TABLE_HALF[1], -TABLE_HALF[2]]
    key = "" if home is None else (f'<keyframe>\n<key name="home" qpos="{f(home[0])}" ctrl="{f(home[1])}"/>\n</keyframe>')
    body_xml = "\n".join(bodies)
    xml = f"""<mujoco model="dexagent_g1_dex1">
<!-- Generated by setup_g1_dex1/build_setup.py from vesoma-models {SOURCE}; edit the generator, not this file. -->
<compiler angle="radian"/>
<option timestep="0.002" impratio="5" integrator="implicitfast"/>
<size nkey="1"/>
<visual>
<global offwidth="1920" offheight="1088"/>
<headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45"/>
</visual>
<default>
<geom solref="0.005" solimp="0.99 0.999 0.0005"/>
</default>
<custom>
<text name="dexagent_setup" data="{HERE.name}"/>
{chr(10).join(arm_custom)}
</custom>
<asset>
<texture name="bg_white" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="8" height="48"/>
@@MESHES@@
</asset>
<worldbody>
<camera name="ego" pos="{f(ego[0])}" quat="{f(ego[1])}" fovy="{f(ego[2])}"/>
<light pos="0 0.35 3.36" dir="0 0 -1" castshadow="false"/>
<light pos="1.72 1.59 2.86" dir="-0.468 -0.338 -0.816" castshadow="false"/>
<body name="table" pos="{f(table_c)}">
<geom name="g_table" type="box" size="{f(TABLE_HALF)}" rgba="0.263 0.204 0.142 1"/>
</body>
{body_xml}
</worldbody>
<contact>
{pairs}
</contact>
<equality>
{welds}
{couplings}
</equality>
<actuator>
{actuators}
</actuator>
{key}
</mujoco>
"""
    meshes = mesh_assets(src, g1_meshes, "assets/g1").replace('<mesh name="', '<mesh name="g1_') + "\n" + \
        mesh_assets(src, dex_meshes, "assets/dex1")
    return xml.replace("@@MESHES@@", meshes), g1_meshes, dex_meshes, mounts


def ego_camera(src: Source):
    """The G1's head camera (the D435i colour camera of the source), in the rig frame: the robot's own view, with the
    real camera's calibration (HEAD_CAM_*) applied."""
    m, d = src.model, src.data
    c = m.camera("camera/head/color").id
    t = src.body("torso_link")
    Rt = d.xmat[t].reshape(3, 3)
    p_t = Rt.T @ (d.cam_xpos[c] - d.xpos[t])
    R_t = Rt.T @ d.cam_xmat[c].reshape(3, 3)
    tp, tR = torso_pose(src)
    pos = tR.apply(p_t) + tp + [0.0, 0.0, HEAD_CAM_DZ]
    R = tR.as_matrix() @ R_t @ Rotation.from_euler("XZ", HEAD_CAM_TURN_DEG, degrees=True).as_matrix()
    return pos, quat_of(R), float(np.degrees(2 * np.arctan(240.0 / HEAD_CAM_FY_PX)))


def write_hand(src: Source, side: str):
    dex = []

    def dex_mesh(n):
        if n not in dex:
            dex.append(n)
        return n

    tree = wrist_chain(side, "", hand_tree(src, side, "", dex_mesh))
    return f"""<mujoco model="dex1_{side}">
<!-- The Unitree Dex1-1 on DexAgent's six-joint virtual wrist (generated by setup_g1_dex1/build_setup.py). The hand
     frame's +z points along the jaws; the jaws close along its y, toward the high end of their range. -->
<compiler angle="radian" meshdir="../assets/dex1/"/>
<asset>
{mesh_assets(src, dex, ".").replace('file="./', 'file="')}
</asset>
<worldbody>
{tree}
</worldbody>
<equality>
{jaw_coupling(side, "")}
</equality>
<actuator>
{hand_actuators(src, side, "")}
</actuator>
</mujoco>
"""


def write_arm(src: Source, side: str):
    g1 = []

    def g1_mesh(n):
        if n not in g1:
            g1.append(n)
        return n

    tree = arm_tree(src, side, lambda n: n, g1_mesh, proposal=False)
    return f"""<mujoco model="g1_arm_{side}">
<!-- The G1's {side} arm, shoulder pitch link to wrist flange (generated by setup_g1_dex1/build_setup.py). Its body
     names are the scene's without the robot_{side}_arm_ prefix; the runtime compiler restores these inertias. -->
<compiler angle="radian" meshdir="../assets/g1/"/>
<asset>
{mesh_assets(src, g1, ".").replace('file="./', 'file="')}
</asset>
<worldbody>
<body name="root">
{tree}
</body>
</worldbody>
</mujoco>
"""


def pretty(xml: str) -> str:
    """Indent generated MJCF two spaces per level (the generator writes flat lines)."""
    out, depth = [], 0
    for line in (ln.strip() for ln in xml.splitlines()):
        if not line:
            continue
        closing = line.startswith("</")
        if closing:
            depth -= 1
        out.append("  " * max(depth, 0) + line)
        if not closing and not line.endswith("/>") and not line.startswith(("<!--", "<?")) and "</" not in line:
            depth += 1
        if line.startswith("<!--") and not line.endswith("-->"):
            depth += 1
        if line.endswith("-->") and not line.startswith("<!--"):
            depth -= 1
    return "\n".join(out) + "\n"


def straight_reach(src):
    m = src.model
    links = sum(float(np.linalg.norm(m.body_pos[src.body(f"left_{s}_link")])) for s in ARM_LINKS[1:])
    return links + float(np.linalg.norm(m.body_pos[src.body("left_dex1_base_link")])) + PAD_CENTRE_X


def write_yaml(src, home_q, model, mounts, ego):
    m = model
    tp, _ = torso_pose(src)
    d = mujoco.MjData(m); d.qpos[:] = home_q; mujoco.mj_kinematics(m, d)
    st = {}
    for side in SIDES:
        arm_q = [float(home_q[m.jnt_qposadr[m.joint(f"robot_{side}_arm_{s}_joint").id]]) for s in ARM_LINKS]
        hand = d.xpos[m.body(f"robot_{side}_{side}_base_yaw").id]
        shoulder = d.xpos[m.body(f"robot_{side}_arm_shoulder_pitch_link").id]
        torque = [float(src.model.jnt_actfrcrange[src.model.joint(f"{side}_{s}_joint").id, 1]) for s in ARM_LINKS]
        armature = [float(src.model.dof_armature[src.model.jnt_dofadr[src.model.joint(f"{side}_{s}_joint").id]]) for s in ARM_LINKS]
        st[side] = dict(arm_q=arm_q, hand=hand.copy(), shoulder=shoulder.copy(), torque=torque, armature=armature)
    cam = {s: wrist_camera(src, s) for s in SIDES}
    R_ego = Rotation.from_quat(np.r_[ego[1][1:], ego[1][0]]).as_matrix()
    look = -R_ego[:, 2]
    pitch = float(np.degrees(np.arcsin(-look[2])))
    yaw = float(np.degrees(np.arctan2(look[1], look[0])))
    focal = 240.0 / np.tan(np.radians(ego[2]) / 2)

    def L(v, n=6):
        return "[" + ", ".join(format(0.0 if abs(float(x)) < 1e-12 else float(x), f".{n}g") for x in v) + "]"

    reach = reach_grid(model, home_q, mounts)

    def span(pts):
        if not pts:
            return "none"
        p = np.asarray(pts)
        return f"{{x: {L([p[:, 0].min(), p[:, 0].max()], 3)}, y: {L([p[:, 1].min(), p[:, 1].max()], 3)}}}"

    both = sorted(set(reach["left"]) & set(reach["right"]))

    stations = ""
    for side in SIDES:
        s = st[side]
        stations += f"""  {side}:
    arm: Unitree G1 {side} arm, 7 DOF (shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw)
    hand: Unitree Dex1-1 parallel-jaw gripper, 2 jaw slides (one motor, coupled), {side}
    arm_xml: robot/g1_arm_{side}.xml
    base_body: robot_{side}_arm_root           # the G1 torso frame; both arms hang from it
    base_pos: {L(tp)}
    base_quat_wxyz: {L(torso_pose(src)[1].as_quat()[[3, 0, 1, 2]])}
    shoulder_pos: {L(s["shoulder"], 5)}
    arm_joints: [{", ".join(f"robot_{side}_arm_{x}_joint" for x in ARM_LINKS)}]
    flange_body: robot_{side}_arm_gripper           # the G1 wrist yaw link
    hand_root_body: robot_{side}_{side}_base_yaw        # the end effector Stage 3 and Stage 4 steer
    hand_xml: robot/dex1_{side}.xml
    wrist_joints: [robot_{side}_{side}_pos_x, robot_{side}_{side}_pos_y, robot_{side}_{side}_pos_z,
                   robot_{side}_{side}_rot_x, robot_{side}_{side}_rot_y, robot_{side}_{side}_rot_z]
    finger_joint_prefix: robot_{side}_{side}_
    jaw_joints: [robot_{side}_{side}_thumb_jaw, robot_{side}_{side}_index_jaw]   # coupled: index = thumb
    home_arm_rad: {L(s["arm_q"])}
    hand_mount_pos: {L(mounts[side][0])}       # the hand frame in the flange frame (the gripper casting's offset)
    hand_mount_quat_wxyz: {L(quat_of(mounts[side][1]))}
"""
    torque = st["left"]["torque"]
    return f"""# DexAgent setup: the Unitree G1's two arms, each with a Unitree Dex1-1 parallel-jaw gripper, at a table.
# Generated by setup_g1_dex1/build_setup.py from vesoma-models {SOURCE}; edit the generator, not this file.
# scene.xml is the floating-wrist planning representation, as in setup/. Physical episodes compile it with
# runtime_robot below: fixed flange-gripper body trees, bounded arm motor torques and the grippers' own jaw servos.
# Units: metres, radians, quaternions as [w, x, y, z]. Angles in degrees say so in the key name.

name: g1_dex1_bench
scene: scene.xml                     # table + the G1 torso and arms + both grippers, keyframe `home`, cameras `ego`, `wrist_left`, `wrist_right`

frame:                               # the rig frame every number below is in; Stage 2 places objects in it
  origin: midpoint of the two shoulder pitch joints, projected onto the tabletop
  x: toward the right arm
  y: into the work, away from the robot
  z: up
  tabletop_z: 0.0

table:
  body: table
  geom: g_table
  center: {L([0.0, TABLE_NEAR_Y + TABLE_HALF[1], -TABLE_HALF[2]])}
  half_extents: {L(TABLE_HALF)}
  work_area:                         # bounds of the {REACH_STEP * 100:g} cm grid points where an arm holds its gripper pointing down with the
                                     # pad centre {REACH_TCP_Z * 100:g} cm above the table (build_setup.reach_grid); the G1 arm is short:
                                     # beyond y = {max(p[1] for s in SIDES for p in reach[s]):.2f} the gripper cannot point straight down
    left: {span(reach["left"])}
    right: {span(reach["right"])}
    both_arms: {span(both)}

stations:                            # one G1 arm + one Dex1-1 each, both on the G1 torso
{stations}
robot:
  body_prefix: robot_                # every robot body, joint and geom starts with this
  attachment: fixed flange child in runtime  # structural attachment with zero relative DOF, never an equality weld
  hand_root_frame: flange in runtime; world in proposal
  hand_frame: +z along the jaws (the approach); jaws close along y
  actuators: 14 arm torque motors and 4 jaw position servos in runtime
  proposal_only: scene.xml virtual wrist joints and soft weld placeholders are for planning, never delivered physics
  nq: {model.nq}
  nu: {model.nu}
  reach: {straight_reach(src):.3f}                       # shoulder pitch axis to the pad centre, sum of the link offsets

runtime_robot:
  schema: dexagent.rigid_robot.v1
  compiler: tool_library.common.rigid_robot.compile_rigid
  attachment: hand base_yaw is a fixed child of the arm's wrist yaw link; no wrist joints
  nq_empty_bench: 18                 # 14 physical arm joints + 4 jaw slides
  nu_empty_bench: 18                 # 14 arm motors + 4 jaw servos
  arm_control: bounded Cartesian impedance mapped through the Jacobian to motor torques, nullspace posture servo
  gravity_compensation: motor feedforward included inside the total motor torque cap
  arm_torque_cap_nm: {L(torque)}
  arm_posture_kp: {L(ARM["kp"])}
  arm_posture_kd: {L(ARM["kd"])}
  arm_params_source: custom fields dexagent_arm_*_<side> in scene.xml (rigid_robot.trusted_arms)
  arm_inertia_source: robot/g1_arm_<side>.xml
  armature_kg_m2: {L(st["left"]["armature"])}
  motor_limit_source: vesoma-models robots/g1/_g1_common.xml actuatorfrcrange (Unitree G1 29-DoF)
  jaw_servo: position, kp {JAW_KP:g} N/m, kv {JAW_KV:g} N s/m, force cap {JAW_FORCE_N:g} N per jaw (vesoma-models _dex1_1.xml)
  jaw_coupling: equality robot_<side>_<side>_jaw_coupling, index_jaw = thumb_jaw (one motor on the real gripper)
  mount_source: flange-to-hand transforms at this scene's home keyframe
  recording: initial mjSTATE_INTEGRATION plus every physics substep state, control and mechanism state

home:                                # keyframe `home` in scene.xml
  keyframe: home
  description: both grippers resting low beside the work, 23 cm out to the sides, 6 cm in front of the shoulders and 8 cm above the table, jaws pointing into the work 10 deg down, jaws half open (46 mm); each arm at least 5 cm off the torso
  hand_root_pos:
    left: {L(st["left"]["hand"], 5)}
    right: {L(st["right"]["hand"], 5)}

cameras:
  ego:                               # the G1's head camera (D435i colour); MuJoCo camera `ego`
    width: 640
    height: 480
    fovy_deg: {ego[2]:.2f}
    focal_px: {focal:.1f}                  # = (height / 2) / tan(fovy / 2); fx = fy, principal point at the image centre
    pos: {L(ego[0], 6)}
    quat_wxyz: {L(ego[1], 6)}   # MuJoCo convention: the camera looks along its -z, y up
    pitch_deg: {pitch:.2f}                 # of the optical axis, below horizontal
    yaw_deg: {yaw:.2f}                   # of the optical axis, from +x; 90 is straight into the work
  wrist_left:                        # rigid to the hand root body; the source's Dex1-1 wrist camera
    parent_body: robot_left_left_base_yaw
    pos: {L(cam["left"][0], 6)}
    quat_wxyz: {L(cam["left"][1], 6)}
    fovy_deg: {cam["left"][2]:.2f}
  wrist_right:
    parent_body: robot_right_right_base_yaw
    pos: {L(cam["right"][0], 6)}
    quat_wxyz: {L(cam["right"][1], 6)}
    fovy_deg: {cam["right"][2]:.2f}

render:                              # Stage 4's Blender materials for the robot (presets of stage4/materials.json); the first match on body/geom/mesh wins
  materials:
    - {{match: "^robot_(left_left|right_right)_", preset: black_plastic}}   # the Dex1-1 grippers and their wrist cameras
    - {{match: "^robot_torso/.*(head|logo)", preset: black_plastic}}
    - {{match: "^robot_", preset: unitree_grey}}                         # the arms and the torso shell

physics:
  timestep: 0.002
  integrator: implicitfast
  impratio: 5
"""


def write_licenses(src, g1, dex, out: Path):
    rows = "\n".join(f"| `{sub}/{src.meshes[n].name}` | `{sha256(src.meshes[n])}` |"
                     for sub, names in (("g1", g1), ("dex1", dex)) for n in names)
    return f"""# Mesh assets

Copied unchanged by `build_setup.py` from the vesoma-models checkout, except the `*_partNN.stl` files: the connected pieces of a
multi-piece visual mesh, split off unchanged (the scene uses one geom per piece) (`robots/g1/assets/visual/` and
`robots/g1_with_dex1/assets/`). The G1 meshes are Unitree's (`unitree_ros`, `mujoco_menagerie/unitree_g1`, BSD-3-Clause);
the Dex1-1 finger and casting meshes are Unitree's Dex1-1 URDF meshes, its pad meshes the simplified collision meshes
that URDF ships; `dex1_wrist_cam_barrel.stl` is written by vesoma-models `scripts/gen_dex1_wrist_cam_barrel.py`.

| file | sha256 |
|---|---|
{rows}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, type=Path, help="the vesoma-models checkout")
    ap.add_argument("--no-images", action="store_true", help="skip rendering images/")
    args = ap.parse_args()
    src = Source(args.models.resolve())
    xml, g1, dex, mounts = write_scene(src)
    for sub in ("robot", "assets/g1", "assets/dex1", "images"):
        (HERE / sub).mkdir(parents=True, exist_ok=True)
    for n in g1:
        shutil.copyfile(src.meshes[n], HERE / "assets/g1" / src.meshes[n].name)
    for n in dex:
        shutil.copyfile(src.meshes[n], HERE / "assets/dex1" / src.meshes[n].name)
    (HERE / "scene.xml").write_text(pretty(xml))
    home_q, home_ctrl, model = solve_home(HERE / "scene.xml", mounts)
    xml, g1, dex, mounts = write_scene(src, (home_q, home_ctrl))
    (HERE / "scene.xml").write_text(pretty(xml))
    for side in SIDES:
        (HERE / f"robot/dex1_{side}.xml").write_text(pretty(write_hand(src, side)))
        (HERE / f"robot/g1_arm_{side}.xml").write_text(pretty(write_arm(src, side)))
    (HERE / "assets/test_cup.xml").write_text(TEST_CUP)
    (HERE / "assets/LICENSES.md").write_text(write_licenses(src, g1, dex, HERE))
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    (HERE / "setup.yaml").write_text(write_yaml(src, home_q, model, mounts, ego_camera(src)))
    for side in SIDES:
        mujoco.MjModel.from_xml_path(str(HERE / f"robot/dex1_{side}.xml"))
        mujoco.MjModel.from_xml_path(str(HERE / f"robot/g1_arm_{side}.xml"))
    print(f"wrote {HERE.relative_to(REPO)}/: scene.xml (nq {model.nq}, nu {model.nu}), setup.yaml, robot/, assets/ "
          f"({len(g1)} G1 + {len(dex)} Dex1-1 meshes)")
    if not args.no_images:
        sys.path.insert(0, str(REPO / "setup"))
        import render_setup
        render_setup.main(HERE)


if __name__ == "__main__":
    main()
