"""Imagen para elegir líneas: zona del frente sombreada + cada candidata como tubo de color con su número.
Uso: blender -b --factory-startup -P render_candidatas.py -- <zona.obj> <candidatas.json> <salida_prefijo> [ids]
"""
import bpy, sys, json, colorsys
from mathutils import Vector
a = sys.argv[sys.argv.index("--") + 1:]
ZONA, CJ, OUT = a[0], a[1], a[2]
SOLO = set(int(x) for x in a[3].split(",")) if len(a) > 3 and a[3] else None
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
V, F = [], []
for l in open(ZONA, encoding="utf-8"):
    if l.startswith("v "): V.append(tuple(float(x) for x in l.split()[1:4]))
    elif l.startswith("f "): F.append([int(x) - 1 for x in l.split()[1:]])
me = bpy.data.meshes.new("z"); me.from_pydata(V, [], F)
ob = bpy.data.objects.new("z", me); sc.collection.objects.link(ob); ob.color = (0.78, 0.78, 0.8, 1)
C = json.load(open(CJ, encoding="utf-8"))
for c in C:
    if SOLO and c["n"] not in SOLO: continue
    cu = bpy.data.curves.new("c%d" % c["n"], "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = 0.6
    sp = cu.splines.new("POLY"); sp.points.add(len(c["puntos"]) - 1)
    for i, p in enumerate(c["puntos"]): sp.points[i].co = (p[0], p[1], p[2] + 0.3, 1)
    o = bpy.data.objects.new("c%d" % c["n"], cu); sc.collection.objects.link(o)
    r, g, b = colorsys.hsv_to_rgb((c["n"] * 0.618) % 1, 0.85, 0.95); o.color = (r, g, b, 1)
    mid = c["puntos"][len(c["puntos"]) // 2]
    td = bpy.data.curves.new("t%d" % c["n"], "FONT"); td.body = str(c["n"]); td.size = 6
    t = bpy.data.objects.new("t%d" % c["n"], td); sc.collection.objects.link(t)
    t.location = (mid[0], mid[1], mid[2] + 3); t.color = (0, 0, 0, 1)
    con = t.constraints.new("TRACK_TO"); con.track_axis = "TRACK_Z"; con.up_axis = "UP_Y"
    t["_mid"] = mid
sc.render.engine = "BLENDER_WORKBENCH"; sh = sc.display.shading; sh.light = "STUDIO"; sh.color_type = "OBJECT"
sh.show_object_outline = True
sc.render.resolution_x = 1800; sc.render.resolution_y = 1300
pts = [Vector(v) for v in V]
mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
cen = (mn + mx) / 2; size = max(mx - mn)
cd = bpy.data.cameras.new("cam"); cd.type = "ORTHO"; cd.ortho_scale = size * 1.15; cd.clip_end = size * 20
cam = bpy.data.objects.new("cam", cd); sc.collection.objects.link(cam); sc.camera = cam
for o in sc.objects:
    if o.name.startswith("t"):
        o.constraints[0].target = cam
for nom, dv in {"34": Vector((1.2, -1.0, 0.9)), "arriba": Vector((0.001, 0, 1)), "frente": Vector((0.25, -1, 0.25)),
                "lado": Vector((1, -0.15, 0.2))}.items():
    dv.normalize(); cam.location = cen + dv * size * 4
    cam.rotation_euler = (cen - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = f"{OUT}_{nom}.png"; bpy.ops.render.render(write_still=True)
print("OK")
