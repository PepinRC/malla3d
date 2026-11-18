"""Render de la SubD final (suave, con crease) + alambre de la jaula, cámara fija por la zona.
Uso: blender -b --factory-startup -P render_subd.py -- <jaula.obj> <salida_prefijo> <zona.obj> [solo_zona=0]"""
import bpy, sys, os, json, bmesh
from mathutils import Vector
a = sys.argv[sys.argv.index("--") + 1:]
OBJ, OUT, ZONA = a[0], a[1], a[2]; SOLO = len(a) > 3 and a[3] == "1"
def leer(p):
    V, F = [], []
    for l in open(p, encoding="utf-8"):
        if l.startswith("v "): V.append(tuple(float(x) for x in l.split()[1:4]))
        elif l.startswith("f "): F.append([int(x.split("/")[0]) - 1 for x in l.split()[1:]])
    return V, F
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
V, F = leer(OBJ); ZV, _ = leer(ZONA)
me = bpy.data.meshes.new("m"); me.from_pydata(V, [], F); me.update()
ob = bpy.data.objects.new("o", me); sc.collection.objects.link(ob); ob.color = (0.8, 0.8, 0.82, 1)
cj = OBJ[:-4] + "_crease.json"
if not SOLO and os.path.exists(cj):
    cr = {tuple(sorted(e)) for e in json.load(open(cj))["crease"]}
    bm = bmesh.new(); bm.from_mesh(me); lay = bm.edges.layers.float.get("crease_edge") or bm.edges.layers.float.new("crease_edge")
    for e in bm.edges:
        if tuple(sorted((e.verts[0].index, e.verts[1].index))) in cr: e[lay] = 1.0
    bm.to_mesh(me); bm.free()
if not SOLO:
    m = ob.modifiers.new("s", "SUBSURF"); m.levels = 3; m.render_levels = 3
    for p in me.polygons: p.use_smooth = True
    w = bpy.data.objects.new("w", me.copy()); sc.collection.objects.link(w)
    wm = w.modifiers.new("wf", "WIREFRAME"); wm.thickness = 0.25; wm.use_replace = True; wm.use_even_offset = False; w.color = (0.05, 0.25, 0.6, 1)
sc.render.engine = "BLENDER_WORKBENCH"; sh = sc.display.shading; sh.light = "STUDIO"; sh.color_type = "OBJECT"; sh.show_cavity = False
sc.render.resolution_x = 1600; sc.render.resolution_y = 1100
pts = [Vector(v) for v in ZV]
mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts))); mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
cen = (mn + mx) / 2; size = max(mx - mn)
cd = bpy.data.cameras.new("c"); cd.type = "ORTHO"; cd.ortho_scale = size * 0.95; cd.clip_end = size * 10
cam = bpy.data.objects.new("c", cd); sc.collection.objects.link(cam); sc.camera = cam
for nom, dv in {"34": Vector((1, -1.1, 0.8))}.items():
    dv.normalize(); cam.location = cen + dv * size * 3
    cam.rotation_euler = (cen - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = f"{OUT}_{nom}.png"; bpy.ops.render.render(write_still=True)
