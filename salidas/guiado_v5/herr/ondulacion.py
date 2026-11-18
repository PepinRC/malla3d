"""Ondulación de la SubD final: en el nivel 3, ángulo entre caras vecinas en las aristas suaves (< 8°, fuera de los
crease). Una superficie limpia cambia de ángulo poco a poco; una con ondas alterna. Reporta la media del cambio de
ángulo entre aristas vecinas (segunda derivada) en grados: menos es más liso.
Uso: blender -b --factory-startup -P ondulacion.py -- <jaula.obj>"""
import bpy, sys, json, bmesh, math
a = sys.argv[sys.argv.index("--") + 1:]; OBJ = a[0]
V, F = [], []
for l in open(OBJ, encoding="utf-8"):
    if l.startswith("v "): V.append(tuple(float(x) for x in l.split()[1:4]))
    elif l.startswith("f "): F.append([int(x.split("/")[0]) - 1 for x in l.split()[1:]])
bpy.ops.wm.read_factory_settings(use_empty=True); sc = bpy.context.scene
me = bpy.data.meshes.new("j"); me.from_pydata(V, [], F); me.update()
cr = {tuple(sorted(e)) for e in json.load(open(OBJ[:-4] + "_crease.json"))["crease"]}
bm = bmesh.new(); bm.from_mesh(me); lay = bm.edges.layers.float.new("crease_edge")
for e in bm.edges:
    if tuple(sorted((e.verts[0].index, e.verts[1].index))) in cr: e[lay] = 1.0
bm.to_mesh(me); bm.free()
ob = bpy.data.objects.new("j", me); sc.collection.objects.link(ob); m = ob.modifiers.new("s", "SUBSURF"); m.levels = 3
ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
b = bmesh.new(); b.from_mesh(ev); b.faces.ensure_lookup_table()
ang = {}
for e in b.edges:
    if len(e.link_faces) == 2:
        x = math.degrees(e.link_faces[0].normal.angle(e.link_faces[1].normal, 0))
        if x < 8: ang[e.index] = x
seg = []
for v in b.verts:
    es = [e for e in v.link_edges if e.index in ang]
    if len(es) == 4:  # vértice regular: pares opuestos de aristas
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                # opuestas = no comparten cara
                if not set(es[i].link_faces) & set(es[j].link_faces):
                    seg.append(abs(ang[es[i].index] - ang[es[j].index]))
seg.sort()
print("ONDULACION media %.3f p95 %.3f grados (n=%d)" % (sum(seg) / len(seg), seg[int(0.95 * len(seg))], len(seg)))
