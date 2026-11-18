"""Mapa de normales: pinta la malla de Tripo según el ángulo entre su cara y nuestra SubD en el punto más cercano.
Saca lo que la distancia esconde: chaflanes perdidos, planos abombados, escalones suavizados.
Uso: blender -b --factory-startup -P mapa_normales.py -- <jaula.obj> <zona.obj> <salida_prefijo> [render=1]
Verde < 5°, amarillo 10°, rojo >= 20°."""
import bpy, sys, json, bmesh, math
from mathutils import Vector
from mathutils.bvhtree import BVHTree
a = sys.argv[sys.argv.index("--") + 1:]
OBJ, ZONA, OUT = a[0], a[1], a[2]; REN = len(a) < 4 or a[3] == "1"
def leer(p):
    V, F = [], []
    for l in open(p, encoding="utf-8"):
        if l.startswith("v "): V.append(tuple(float(x) for x in l.split()[1:4]))
        elif l.startswith("f "): F.append([int(x.split("/")[0]) - 1 for x in l.split()[1:]])
    return V, F
bpy.ops.wm.read_factory_settings(use_empty=True); sc = bpy.context.scene
V, F = leer(OBJ); me = bpy.data.meshes.new("j"); me.from_pydata(V, [], F); me.update()
try: cr = {tuple(sorted(e)) for e in json.load(open(OBJ[:-4] + "_crease.json"))["crease"]}
except Exception: cr = set()
bm = bmesh.new(); bm.from_mesh(me); lay = bm.edges.layers.float.new("crease_edge")
for e in bm.edges:
    if tuple(sorted((e.verts[0].index, e.verts[1].index))) in cr: e[lay] = 1.0
bm.to_mesh(me); bm.free()
ob = bpy.data.objects.new("j", me); sc.collection.objects.link(ob); m = ob.modifiers.new("s", "SUBSURF"); m.levels = 4
ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
bvh = BVHTree.FromPolygons([v.co.copy() for v in ev.vertices], [list(p.vertices) for p in ev.polygons])
ZV, ZF = leer(ZONA)
zm = bpy.data.meshes.new("z"); zm.from_pydata(ZV, [], ZF); zm.update()
bz = bmesh.new(); bz.from_mesh(zm); bmesh.ops.triangulate(bz, faces=bz.faces[:])
for _ in range(4): bmesh.ops.subdivide_edges(bz, edges=bz.edges[:], cuts=1, use_grid_fill=True)
bz.to_mesh(zm); bz.free(); zm.update()
att = zm.color_attributes.new("n", "FLOAT_COLOR", "CORNER")
A = []
for p in zm.polygons:
    loc, n, i, d = bvh.find_nearest(p.center)
    ang = math.degrees(math.acos(min(1.0, abs(p.normal.dot(n))))) if n is not None else 90
    A.append((ang, p.area))
    t = min(ang / 20.0, 1.0)
    c = (0.15 + 0.8 * t, 0.65 * (1 - t) + 0.15, 0.25 * (1 - t), 1) if t < 1 else (0.85, 0.1, 0.1, 1)
    for li in p.loop_indices: att.data[li].color = c
tot = sum(x[1] for x in A)
print("NORMALES area>5° %.1f%%  >10° %.1f%%  >20° %.1f%%  media %.2f°" % tuple([100 * sum(ar for g, ar in A if g > u) / tot for u in (5, 10, 20)] + [sum(g * ar for g, ar in A) / tot]))
if REN:
    zo = bpy.data.objects.new("z", zm); sc.collection.objects.link(zo); sc.collection.objects.unlink(ob)
    sc.render.engine = "BLENDER_WORKBENCH"; sh = sc.display.shading; sh.light = "STUDIO"; sh.color_type = "VERTEX"
    sc.render.resolution_x = 1600; sc.render.resolution_y = 1100
    pts = [Vector(v) for v in ZV]
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts))); mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    cen = (mn + mx) / 2; size = max(mx - mn)
    cd = bpy.data.cameras.new("c"); cd.type = "ORTHO"; cd.ortho_scale = size * 0.95; cd.clip_end = size * 10
    cam = bpy.data.objects.new("c", cd); sc.collection.objects.link(cam); sc.camera = cam
    dv = Vector((1, -1.1, 0.8)); dv.normalize(); cam.location = cen + dv * size * 3
    cam.rotation_euler = (cen - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = OUT + "_34.png"; bpy.ops.render.render(write_still=True)
