"""Devuelve los chaflanes de Tripo a la jaula: cada arista viva que en Tripo es un chaflán angosto (dos dobleces
paralelos a 0.5–4 mm) se bisela en la jaula y las dos orillas nuevas se llevan a los dobleces reales de Tripo.
Las orillas del chaflán llevan crease. Uso:
blender -b --factory-startup -P chaflanes.py -- <jaula.obj> <lineas_sinfundir.json> <salida.obj> [ancho_def=1.8]"""
import bpy, bmesh, sys, json, math
from mathutils import Vector
from mathutils.kdtree import KDTree
a = sys.argv[sys.argv.index("--") + 1:]
OBJ, LJ, OUT = a[0], a[1], a[2]; ANCHO = float(a[3]) if len(a) > 3 else 1.8
V, F = [], []
for l in open(OBJ, encoding="utf-8"):
    if l.startswith("v "): V.append(tuple(float(x) for x in l.split()[1:4]))
    elif l.startswith("f "): F.append([int(x.split("/")[0]) - 1 for x in l.split()[1:]])
cr = {tuple(sorted(e)) for e in json.load(open(OBJ[:-4] + "_crease.json"))["crease"]}
# puntos densos de las líneas sin fundir, con su tangente
P, T = [], []
for ln in json.load(open(LJ, encoding="utf-8")):
    Q = [Vector(p) for p in ln["puntos"]]
    for k in range(len(Q) - 1):
        d = Q[k + 1] - Q[k]; L = d.length
        if L < 1e-9: continue
        n = max(1, int(L / 0.2))
        for s in range(n): P.append(Q[k] + d * (s / n)); T.append(d / L)
kd = KDTree(len(P))
for i, p in enumerate(P): kd.insert(p, i)
kd.balance()
bm = bmesh.new()
vs = [bm.verts.new(v) for v in V]
for f in F:
    try: bm.faces.new([vs[i] for i in f])
    except ValueError: pass
bm.verts.index_update(); bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table(); bm.normal_update()
lay = bm.edges.layers.float.new("crease_edge")
for e in bm.edges:
    if tuple(sorted((e.verts[0].index, e.verts[1].index))) in cr: e[lay] = 1.0
cham = []; anchos = []
for e in bm.edges:
    if e[lay] < 0.5 or len(e.link_faces) != 2: continue
    m = (e.verts[0].co + e.verts[1].co) / 2; t = (e.verts[1].co - e.verts[0].co).normalized()
    n = (e.link_faces[0].normal + e.link_faces[1].normal).normalized(); b = n.cross(t)
    izq = der = None
    for (p, i, d) in kd.find_range(m, 3.0):
        if abs(T[i].dot(t)) < 0.8: continue
        lat = (p - m).dot(b)
        if 0.25 < lat < 2.5: izq = lat if izq is None else min(izq, lat)
        elif -2.5 < lat < -0.25: der = -lat if der is None else min(der, -lat)
    if izq is not None and der is not None and 0.6 < izq + der < 4.0:
        cham.append(e); anchos.append(izq + der)
anchos.sort(); w = anchos[len(anchos) // 2] if anchos else ANCHO
print("CHAFLANES aristas", len(cham), "de", sum(1 for e in bm.edges if e[lay] > 0.5), "ancho mediano %.2f" % w)
viejos = set(bm.verts)
res = bmesh.ops.bevel(bm, geom=cham + list({v for e in cham for v in e.verts}), offset=w / 2, offset_type="OFFSET",
                      segments=1, profile=0.5, affect="EDGES", clamp_overlap=True)
caras_bisel = set(res["faces"])
bm.verts.ensure_lookup_table(); bm.normal_update()
# llevar los vértices nuevos al doblez real más cercano
nuevos = [v for v in bm.verts if v not in viejos]
mov = 0
for v in nuevos:
    p, i, d = kd.find(v.co)
    if d < 1.5: v.co = p; mov += 1
# crease: orillas del chaflán (entre cara de bisel y cara normal) + las aristas vivas que no se biselaron
for e in bm.edges:
    fb = [f in caras_bisel for f in e.link_faces]
    if len(fb) == 2 and fb[0] != fb[1]: e[lay] = 1.0
    elif len(fb) == 2 and fb[0] and fb[1]: e[lay] = 0.0
bm.verts.ensure_lookup_table(); bm.verts.index_update()
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("# jaula con chaflanes\n")
    for v in bm.verts: fh.write("v %.6f %.6f %.6f\n" % tuple(v.co))
    for f in bm.faces: fh.write("f " + " ".join(str(v.index + 1) for v in f.verts) + "\n")
json.dump({"crease": [[e.verts[0].index, e.verts[1].index] for e in bm.edges if e[lay] > 0.5]}, open(OUT[:-4] + "_crease.json", "w"))
print("CHAFLANES caras", len(bm.faces), "bisel", len(caras_bisel), "nuevos", len(nuevos), "llevados al doblez", mov,
      "no cuadradas", sum(1 for f in bm.faces if len(f.verts) != 4))
