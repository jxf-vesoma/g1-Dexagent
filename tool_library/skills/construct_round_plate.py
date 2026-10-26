TOOL = {
    "name": "construct_round_plate",
    "title": "Construct round plate or coaster",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "Builds a thin circular plate, coaster or puck as one rigid cylindrical body with its origin at the bottom centre.",
    "description": "Creates a metric MJCF asset for a flat round rigid support. Radius, thickness, colour, mass, friction, placement and fixed/free status are parameters. The colliding top face is planar, so another rigid object can rest on it. The body frame is at the bottom centre, making table staging unambiguous.",
    "inputs": [
        {"name":"out_dir","type":"path","description":"Folder receiving XML and params JSON."},
        {"name":"name","type":"str","description":"Asset and body name."},
        {"name":"radius","type":"float","description":"Outer radius in metres."},
        {"name":"thickness","type":"float","description":"Thickness in metres."},
        {"name":"rgba","type":"list[4]","description":"Surface colour."},
        {"name":"mass","type":"float","description":"Mass in kg."},
        {"name":"friction","type":"list[3]","description":"MuJoCo friction triple."},
        {"name":"pos","type":"list[3]","description":"Bottom-centre position."},
        {"name":"yaw_deg","type":"float","description":"Yaw about world z."},
        {"name":"fixed","type":"bool","description":"Whether the body is fixed."},
        {"name":"include_floor","type":"bool","description":"Add a standalone floor."}
    ],
    "outputs": [
        {"name":"xml","type":"path","description":"MJCF path."},
        {"name":"params_json","type":"path","description":"Built parameters."},
        {"name":"body","type":"str","description":"Root body name."},
        {"name":"joint","type":"str | None","description":"Free joint name unless fixed."},
        {"name":"radius_m","type":"float","description":"Built radius."},
        {"name":"thickness_m","type":"float","description":"Built thickness."},
        {"name":"mass_kg","type":"float","description":"Built mass."}
    ],
    "status":"tested",
    "entrypoint":"construct_round_plate",
    "requirements":["mujoco"],
    "tags":["plate","coaster","disc","puck","rigid","support"],
    "used_in":["Pick up the cup and place it on the plate"],
    "cli":"python -m tool_library.skills.construct_round_plate OUT_DIR"
}

from pathlib import Path
import json, math

DEFAULTS = dict(name="plate", radius=0.04, thickness=0.004, rgba=[0.18,0.22,0.17,1.0], mass=0.04,
                friction=[1.0,0.02,0.002], pos=[0,0,0], yaw_deg=0.0, fixed=False, include_floor=True)

def construct_round_plate(out_dir, **params):
    unknown=set(params)-set(DEFAULTS)
    if unknown: raise ValueError(f"unknown parameters: {sorted(unknown)}")
    p=DEFAULTS|params
    if p['radius']<=0: raise ValueError("radius must be positive")
    if p['thickness']<=0: raise ValueError("thickness must be positive")
    if p['mass']<=0: raise ValueError("mass must be positive")
    if len(p['rgba'])!=4: raise ValueError("rgba must have four values")
    if len(p['friction'])!=3: raise ValueError("friction must have three values")
    if len(p['pos'])!=3: raise ValueError("pos must have three values")
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    name=str(p['name']); q=math.cos(math.radians(float(p['yaw_deg']))/2); s=math.sin(math.radians(float(p['yaw_deg']))/2)
    rgba=' '.join(map(str,p['rgba'])); fr=' '.join(map(str,p['friction'])); pos=' '.join(map(str,p['pos']))
    joint='' if p['fixed'] else f'<freejoint name="{name}_free"/>'
    floor='<geom name="floor" type="plane" size="1 1 0.01" rgba="0.85 0.85 0.85 1"/>' if p['include_floor'] else ''
    xml=f'''<mujoco model="{name}"><option timestep="0.002"/><worldbody>{floor}<body name="{name}" pos="{pos}" quat="{q} 0 0 {s}">{joint}<geom name="{name}_geom" type="cylinder" size="{p['radius']} {p['thickness']/2}" pos="0 0 {p['thickness']/2}" rgba="{rgba}" mass="{p['mass']}" friction="{fr}" condim="6"/><site name="{name}_top" pos="0 0 {p['thickness']}" size="0.002" rgba="0 0 0 0"/></body></worldbody></mujoco>'''
    xp=out/f'{name}.xml'; xp.write_text(xml)
    pp=out/f'{name}_params.json'; pp.write_text(json.dumps(p,indent=2))
    return {"xml":str(xp),"params_json":str(pp),"body":name,"joint":None if p['fixed'] else f"{name}_free","geoms":[f"{name}_geom"],"sites":{"top":f"{name}_top"},"radius_m":float(p['radius']),"thickness_m":float(p['thickness']),"mass_kg":float(p['mass'])}

if __name__=='__main__':
    import sys
    print(json.dumps(construct_round_plate(sys.argv[1]),indent=2))
