"""Render del auto completo, con la misma cámara para Tripo y para nuestras jaulas.
Uso: blender -b --factory-startup -P render_auto.py -- <modo tripo|jaula> <salida_prefijo> [carpetas separadas por ;]
modo jaula: cada pieza panel/tubo usa la primera carpeta que tenga <id>.obj (+ _crease.json); las piezas espejo se
reflejan en X. SubD nivel 2 con crease y el alambre de la jaula encima. La carrocería (s054) va en otro tono.
modo tripo: las piezas de la Malla Smart tal cual (facetadas). Sin ruedas ni astillas."""
import bpy, bmesh, sys, os, json
from mathutils import Vector
a = sys.argv[sys.argv.index("--") + 1:]
MODO, OUT = a[0], a[1]; CARP = a[2].split(";") if len(a) > 2 else []
import os as _os
RAIZ = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
inv = json.load(open(RAIZ + "/datos/smart_v5/piezas.json", encoding="utf-8"))["piezas"]
bpy.ops.wm.read_factory_settings(use_empty=True); sc = bpy.context.scene
def leer(p):
    V, F = [], []
    for l in open(p, encoding="utf-8"):
        if l.startswith("v "): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("f "): F.append([int(x.split("/")[0]) - 1 for x in l.split()[1:]])
    return V, F
def buscar(pid):
    for c in CARP:
        if os.path.exists(os.path.join(c, pid + ".obj")): return os.path.join(c, pid + ".obj")
    return None
todos = []; faltan = []; ncaras = 0
for p in inv:
    if p["clase"] not in ("panel", "tubo"): continue
    if MODO == "tripo":
        V, F = leer(RAIZ + "/datos/smart_v5/piezas/%s.obj" % p["id"]); cr = set()
    else:
        src = p["id"] if p["usar"] else p["espejo_de"]
        ruta = buscar(src)
        if not ruta: faltan.append(p["id"]); continue
        V, F = leer(ruta)
        try: cr = {tuple(sorted(e)) for e in json.load(open(ruta[:-4] + "_crease.json"))["crease"]}
        except Exception: cr = set()
        if not p["usar"]:
            V = [[-x, y, z] for x, y, z in V]; F = [f[::-1] for f in F]
    ncaras += len(F)
    me = bpy.data.meshes.new(p["id"]); me.from_pydata([tuple(v) for v in V], [], F); me.update()
    ob = bpy.data.objects.new(p["id"], me); sc.collection.objects.link(ob); todos += [Vector(v) for v in V]
    ob.color = (0.82, 0.82, 0.84, 1) if p["id"] != "s054" else (0.72, 0.78, 0.86, 1)
    if MODO == "jaula":
        bm = bmesh.new(); bm.from_mesh(me); lay = bm.edges.layers.float.new("crease_edge"); bm.verts.index_update()
        for e in bm.edges:
            if tuple(sorted((e.verts[0].index, e.verts[1].index))) in cr: e[lay] = 1.0
        bm.to_mesh(me); bm.free()
        for pl in me.polygons: pl.use_smooth = True
        m = ob.modifiers.new("s", "SUBSURF"); m.levels = 2; m.render_levels = 2
        w = bpy.data.objects.new(p["id"] + "_w", me.copy()); sc.collection.objects.link(w)
        wm = w.modifiers.new("wf", "WIREFRAME"); wm.thickness = 0.35; wm.use_replace = True; wm.use_even_offset = False
        w.color = (0.05, 0.25, 0.6, 1)
# cámara fija: caja de las piezas de Tripo, para que las tres columnas coincidan
todos = []
for p in inv:
    if p["clase"] in ("panel", "tubo"):
        Vt, _ = leer(RAIZ + "/datos/smart_v5/piezas/%s.obj" % p["id"]); todos += [Vector(v) for v in Vt]
sc.render.engine = "BLENDER_WORKBENCH"; sh = sc.display.shading; sh.light = "STUDIO"; sh.color_type = "OBJECT"
sc.render.resolution_x = 1600; sc.render.resolution_y = 1000
mn = Vector((min(v.x for v in todos), min(v.y for v in todos), min(v.z for v in todos))); mx = Vector((max(v.x for v in todos), max(v.y for v in todos), max(v.z for v in todos)))
cen = (mn + mx) / 2; size = (mx - mn).length
cd = bpy.data.cameras.new("c"); cd.type = "ORTHO"; cd.clip_end = size * 10
cam = bpy.data.objects.new("c", cd); sc.collection.objects.link(cam); sc.camera = cam
vistas = {"frente34": ((1, -1.1, 0.55), 0.95), "lado": ((1, 0, 0.02), 0.95), "atras34": ((1, 1.1, 0.55), 0.95), "arriba": ((0.0001, 0.0001, 1), 0.95)}
for nom, (dv, s) in vistas.items():
    dv = Vector(dv).normalized(); cam.location = cen + dv * size * 2; cd.ortho_scale = size * s
    cam.rotation_euler = (cen - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = "%s_%s.png" % (OUT, nom); bpy.ops.render.render(write_still=True)
print("AUTO", MODO, "caras", ncaras, "faltan", faltan)
