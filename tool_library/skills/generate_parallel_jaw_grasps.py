TOOL = {
    "name": "generate_parallel_jaw_grasps",
    "title": "Generate parallel-jaw grasps for a rigid object",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Enumerate antipodal grasps for a two-jaw gripper (the Dex1-1 of setup_g1_dex1): jaw axes through the object whose two surface points face each other inside the friction cone and fit the opening, the gripper placed so its pads close on them, screened against the table and verified by grasp_lift_test.",
    "description": "For a hand whose fingers are two jaw slides (hand.hand_spec_for finds <side>_thumb_jaw and <side>_index_jaw), samples jaw axes (a Fibonacci half-sphere plus the world and object axes) through lines at several depths along each approach and heights above the centre of mass. Each line is ray cast against the grasped object from both sides: the two first hits are the jaw contacts, their surface normals must lie within cone_deg of the jaw axis (antipodal), and their distance must fit the jaw opening with clearance_m to spare on each side. Approaches are sampled around the jaw axis (perpendicular to it, never from below the table: approach z at most max_approach_up). The gripper is placed so the midpoint of its pad centres (the tip sites) sits on the midpoint of the contacts, the jaws closing along the hand's y. Static screen: the OPEN gripper at the grasp and on its retreat along -approach clears the object and the table, and the gripper with its jaws ON the contacts (finger_targets) does not enter the table (a closure the table blocks, such as a jaw under a box resting on it, is rejected). Survivors are ranked by the grasp style first (`style`, or the one the Stage 1 subgoal grasping this object describes when `scene_semantic_json` is given: 'side' puts grasps whose retreat rises less than 70 deg first, nearest a 45 deg tilt, which keeps the gripper body off the table; 'top' puts grasps retreating within 20 deg of vertical first, most vertical first; no style has no orientation preference), then most antipodal, then nearest the object's centre; with verify=True the top verify_top are driven through grasp_lift_test and the ones that lift come first. A part wider than the opening along every antipodal axis returns no grasp.",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor}; the last object is grasped; hand_xml a parallel-jaw hand (setup_g1_dex1/robot/dex1_<side>.xml)"},
        {"name": "n_axes", "type": "int", "description": "jaw axes sampled on the half-sphere, besides the world and object axes (default 24)"},
        {"name": "approach_step_deg", "type": "float", "description": "step of the approach direction around each jaw axis (default 30)"},
        {"name": "depths_m", "type": "list of float", "description": "offsets of the contact line from the object's centre of mass along the approach, negative toward the gripper (default [0, -0.02, -0.04, 0.02])"},
        {"name": "heights_m", "type": "list of float", "description": "offsets of the contact line from the centre of mass along world z (default [0, 0.02, 0.04]): a grasp high on a tall object leaves room under a tilted or horizontal gripper; lines that leave the object miss"},
        {"name": "cone_deg", "type": "float", "description": "largest angle between a contact normal and the jaw axis (default 20; a friction coefficient of 0.36)"},
        {"name": "clearance_m", "type": "float", "description": "opening to spare on each side of the part (default 0.004)"},
        {"name": "max_approach_up", "type": "float", "description": "largest upward component of the approach direction (default 0.05: horizontal or from above)"},
        {"name": "retreat_m", "type": "float", "description": "how far the open gripper must retreat clear along -approach (default 0.09)"},
        {"name": "style", "type": "str | None", "description": "'side', 'top' or None: the grasp style to rank first (default None: read from scene_semantic_json when given, else no preference). Matching grasps come first; the others still follow"},
        {"name": "scene_semantic_json", "type": "str | None", "description": "Stage 1 scene_semantics.json: when `style` is None, the style is read from the action and acting-hand role of the subgoal that grasps this object (`subgoal`, or the first grasp_lift / grasp subgoal naming the scene's `target`): 'from above', 'top-down', 'from the top' give 'top'; 'side', 'around', 'wrap' give 'side'"},
        {"name": "subgoal", "type": "int | None", "description": "the Stage 1 subgoal index to read the style from (default: found from the scene's target)"},
        {"name": "verify", "type": "bool", "description": "run grasp_lift_test on the top verify_top candidates (default True)"},
        {"name": "verify_top", "type": "int", "description": "how many candidates to verify (default 4)"},
        {"name": "max_candidates", "type": "int", "description": "how many ranked candidates to return (default 20)"},
        {"name": "out_json", "type": "str | None", "description": "write the result here as JSON (its `grasps` entries are poses video_guided_approach, optimize_grasp_pose and the executor take)"},
    ],
    "outputs": [
        {"name": "grasps", "type": "list of pose dicts", "description": "ranked: {side, pos, euler, finger_targets (jaws on the contacts), finger_targets_open, grip_fingers, family 'parallel_jaw', jaw_axis, approach (the retreat direction, outward from the object, as hand.approach_axis reads it), contacts, width_m, antipodal_deg, depth_m, height_m, elevation_deg (how far the retreat rises above horizontal), style ('side' or 'top'), lift (if verified)}; with verify only the ones that lift"},
        {"name": "n_candidates / n_feasible / n_verified / n_lift", "type": "int", "description": "lines tried, statically feasible grasps, verified, lifted"},
        {"name": "rejected", "type": "dict", "description": "count per reason: miss, too_wide, not_antipodal, collision, retreat, table_blocks_closure"},
        {"name": "opening_m", "type": "float", "description": "the gripper's full opening between the pads"},
        {"name": "style / style_source / n_side / n_top", "type": "str | int", "description": "the style ranked first (None when none), where it came from ('argument', 'scene_semantic_json subgoal <i>: <text>', or None), and how many feasible grasps are of each style"},
    ],
    "status": "tested",
    "entrypoint": "generate_parallel_jaw_grasps",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "candidate-generation", "parallel-jaw", "antipodal"],
    "used_in": [],
}

import warnings

import numpy as np

_TOL_MM = 0.5          # interpenetration the static screen tolerates (the pad sitting on the surface)
_TOP_MIN_ELEV_DEG = 70.0   # a grasp whose retreat rises at least this much above horizontal is a top grasp
_SIDE_TILT_DEG = 45.0      # the side grasps ranked first: tilted, the gripper body clear of the table
_TOP_WORDS = ("from above", "top-down", "top down", "from the top", "over the top", "overhead")
_SIDE_WORDS = ("side", "around", "wrap", "encircl", "sideways")


def style_from_semantics(scene_semantic_json, target=None, subgoal=None):
    """(style, source) from the Stage 1 subgoal that grasps `target`: its action and the acting hand's role, searched
    for top words first ('grasps the rim from above' is a top grasp), then side words. (None, None) when silent."""
    import json
    from pathlib import Path
    sem = json.loads(Path(scene_semantic_json).read_text())
    subs = sem.get("subgoals") or []
    if subgoal is not None:
        subs = [s for s in subs if s.get("index") == subgoal]
    else:
        subs = [s for s in subs if "grasp" in str(s.get("motion", "")) and (target is None or not s.get("object_refs")
                                                                          or target in s.get("object_refs"))]
    for s in subs[:1]:
        role = (s.get("hand_roles") or {}).get(s.get("acting_hand"), "")
        text = f"{s.get('action', '')} {role}".strip()
        low = text.lower()
        for style, words in (("top", _TOP_WORDS), ("side", _SIDE_WORDS)):
            if any(w in low for w in words):
                return style, f"scene_semantic_json subgoal {s.get('index')}: {text}"
    return None, None


def _fibonacci_half_sphere(n):
    """n unit directions spread over the half-sphere x > 0 or (x = 0, y >= 0): one per jaw axis (u and -u are the
    same axis)."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    v = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return [_canon(u) for u in v]


def _canon(u):
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    for k in range(3):
        if abs(u[k]) > 1e-9:
            return u if u[k] > 0 else -u
    return u


def _ray_hit(model, data, start, direction, gids):
    """First hit of the ray on one of `gids`, ignoring every other geom: (point, outward normal) or None.
    Each geom is intersected on its own: a visual mesh coinciding with the collision surface (construct_cup's shell
    over its wedges) would otherwise put the ray inside the collision geom, where mj_ray reports no hit."""
    import mujoco
    p = np.asarray(start, float)
    v = np.asarray(direction, float)
    best, nrm = None, np.zeros(3)
    for g in gids:
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            dist = mujoco.mj_rayMesh(model, data, int(g), p, v, nrm)
        else:
            dist = mujoco.mju_rayGeom(data.geom_xpos[g], data.geom_xmat[g], model.geom_size[g], p, v,
                                      int(model.geom_type[g]), nrm)
        if dist >= 0 and (best is None or dist < best[0]):
            best = (dist, nrm.copy())
    if best is None:
        return None
    n = best[1] / max(np.linalg.norm(best[1]), 1e-12)
    return p + best[0] * v, (n if n @ v <= 0 else -n)


def _jaw_geometry(model, rig):
    """In the hand root frame: the midpoint of the pad centres at jaws 0, the pad gap there, and how fast the gap
    closes per unit of jaw value (both jaws together, as the coupling drives them)."""
    import mujoco
    d = mujoco.MjData(model)
    jaws = list(rig.finger_range)

    def tips(q):
        for k in jaws:
            d.qpos[rig.finger_qpos[k]] = q
        mujoco.mj_kinematics(model, d)
        R = d.xmat[rig.root_body].reshape(3, 3)
        return [R.T @ (d.site_xpos[s] - d.xpos[rig.root_body]) for s in rig.fingertip_sites.values()]

    a, b = tips(0.0)
    gap0 = float(np.linalg.norm(a - b))
    a1, b1 = tips(0.01)
    rate = (gap0 - float(np.linalg.norm(a1 - b1))) / 0.01
    return (a + b) / 2.0, gap0, rate


def generate_parallel_jaw_grasps(scene, n_axes=24, approach_step_deg=30.0, depths_m=(0.0, -0.02, -0.04, 0.02),
                                 heights_m=(0.0, 0.02, 0.04), cone_deg=20.0, clearance_m=0.004, max_approach_up=0.05, retreat_m=0.09,
                                 style=None, scene_semantic_json=None, subgoal=None,
                                 verify=True, verify_top=4, max_candidates=20, out_json=None):
    import mujoco
    from scipy.spatial.transform import Rotation
    from tool_library.common import hand as H
    style_source = "argument" if style is not None else None
    if style is None and scene_semantic_json:
        import json
        from pathlib import Path
        sc = scene if isinstance(scene, dict) else json.loads(Path(scene).read_text())
        style, style_source = style_from_semantics(scene_semantic_json, sc.get("target"), subgoal)
    if style not in (None, "side", "top"):
        raise ValueError(f"style must be 'side', 'top' or None, not {style!r}")
    model, data0, rig, ids = H.open_scene(scene)
    roles = H.jaw_roles(rig)
    if len(roles) != 2:
        raise ValueError("generate_parallel_jaw_grasps needs a two-jaw gripper (<side>_thumb_jaw and <side>_index_jaw)")
    jaws = list(rig.finger_range)
    lo = min(rig.finger_range[k][0] for k in jaws)
    hi = max(rig.finger_range[k][1] for k in jaws)
    tcp_local, gap0, rate = _jaw_geometry(model, rig)
    opening = gap0 - rate * lo
    bid = H.grasp_body(model, scene, ids)
    gids = set(ids["grasp_gids"])
    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos
    mujoco.mj_forward(model, d)
    com = d.xipos[bid].copy()
    reach = 2.0 * max(float(model.geom_rbound[g]) + float(np.linalg.norm(d.geom_xpos[g] - com)) for g in gids) + 0.05
    axes = _fibonacci_half_sphere(int(n_axes))
    axes += [_canon(v) for v in np.eye(3)] + [_canon(v) for v in d.xmat[bid].reshape(3, 3).T]
    uniq = []
    for u in axes:
        if all(abs(u @ w) < 0.999 for w in uniq):
            uniq.append(u)
    cos_cone = np.cos(np.radians(cone_deg))
    rejected = {k: 0 for k in ("miss", "too_wide", "not_antipodal", "collision", "retreat", "table_blocks_closure")}
    cands, n_lines = [], 0

    def place(pos, eul, q):
        d.qpos[:] = data0.qpos
        rig.teleport(d, pos, eul)
        for k in jaws:
            d.qpos[rig.finger_qpos[k]] = q
        mujoco.mj_forward(model, d)

    for u in uniq:
        e1 = np.cross(u, [0.0, 0.0, 1.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(u, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        for th in np.radians(np.arange(0.0, 360.0, approach_step_deg)):
            a = np.cos(th) * e1 + np.sin(th) * e2
            if a[2] > max_approach_up:
                continue
            R = np.column_stack([np.cross(u, a), u, a])
            with warnings.catch_warnings():          # gimbal lock: the angles returned still give R exactly
                warnings.simplefilter("ignore", UserWarning)
                eul = Rotation.from_matrix(R).as_euler("XYZ")
            for s, hz in ((s, hz) for s in depths_m for hz in heights_m):
                n_lines += 1
                c = com + float(s) * a + np.array([0.0, 0.0, float(hz)])
                h1 = _ray_hit(model, data0, c + reach * u, -u, gids)
                h2 = _ray_hit(model, data0, c - reach * u, u, gids)
                if h1 is None or h2 is None:
                    rejected["miss"] += 1
                    continue
                (p1, n1), (p2, n2) = h1, h2
                width = float((p1 - p2) @ u)
                if width + 2.0 * clearance_m > opening:
                    rejected["too_wide"] += 1
                    continue
                ang = float(np.degrees(np.arccos(np.clip(min(n1 @ u, -(n2 @ u)), -1.0, 1.0))))
                if min(n1 @ u, -(n2 @ u)) < cos_cone:
                    rejected["not_antipodal"] += 1
                    continue
                mid = (p1 + p2) / 2.0
                pos = mid - R @ tcp_local
                q_touch = float(np.clip((gap0 - width) / rate, lo, hi))
                place(pos, eul, lo)
                cl = H.clearance(model, d, rig, ids)
                if cl["object_mm"] > _TOL_MM or cl["floor_mm"] > _TOL_MM:
                    rejected["collision"] += 1
                    continue
                clear = True
                for k in (1, 2, 3):
                    place(pos - a * retreat_m * k / 3.0, eul, lo)
                    cl = H.clearance(model, d, rig, ids)
                    if cl["object_mm"] > _TOL_MM or cl["floor_mm"] > _TOL_MM:
                        clear = False
                        break
                if not clear:
                    rejected["retreat"] += 1
                    continue
                place(pos, eul, q_touch)
                cl = H.clearance(model, d, rig, ids, ignore_tips=rig.fingers)
                if cl["floor_mm"] > _TOL_MM or cl["object_mm"] > _TOL_MM:
                    rejected["table_blocks_closure"] += 1
                    continue
                elev = float(np.degrees(np.arcsin(np.clip(-a[2], -1.0, 1.0))))
                cands.append({"side": rig.side, "pos": [float(x) for x in pos], "euler": [float(x) for x in eul],
                              "finger_targets": {k: q_touch for k in jaws},
                              "finger_targets_open": {k: float(lo) for k in jaws},
                              "grip_fingers": list(rig.fingers), "family": "parallel_jaw",
                              "jaw_axis": [round(float(x), 6) for x in u], "approach": [round(float(x), 6) for x in -a],
                              "contacts": [[round(float(x), 5) for x in p1], [round(float(x), 5) for x in p2]],
                              "width_m": round(width, 5), "antipodal_deg": round(ang, 2), "depth_m": float(s),
                              "height_m": float(hz), "elevation_deg": round(elev, 1),
                              "style": "top" if elev >= _TOP_MIN_ELEV_DEG else "side"})

    def rank(g):
        if style == "top":
            first, turn = g["style"] != "top", -g["elevation_deg"]
        elif style == "side":
            first, turn = g["style"] != "side", abs(g["elevation_deg"] - _SIDE_TILT_DEG)
        else:
            first, turn = False, 0.0
        return (first, round(g["antipodal_deg"] / 5.0), round(turn / 15.0), abs(g["depth_m"]), abs(g["height_m"]))

    cands.sort(key=rank)
    res = {"n_candidates": n_lines, "n_feasible": len(cands), "rejected": rejected, "opening_m": round(opening, 5),
           "n_verified": 0, "n_lift": 0, "style": style, "style_source": style_source,
           "n_side": sum(g["style"] == "side" for g in cands), "n_top": sum(g["style"] == "top" for g in cands)}
    if verify:
        from tool_library.verifiers.grasp_lift_test import grasp_lift_test
        lifted = []
        for g in cands[:int(verify_top)]:
            r = grasp_lift_test(scene, g)
            g["lift"] = {"ok": r["ok"], "lifts": r["lifts"], "first_fail": r["first_fail"],
                         "force_N": r["grip"]["force_N"], "penetration_mm": r["penetration"]["max_contact_mm"]}
            res["n_verified"] += 1
            if r["ok"]:
                lifted.append(g)
        res["n_lift"] = len(lifted)
        res["grasps"] = lifted[:int(max_candidates)]
    else:
        res["grasps"] = cands[:int(max_candidates)]
    if out_json:
        import json
        from pathlib import Path
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(res, indent=1, default=float))
        res["out_json"] = str(out_json)
    return res
