#! python3
"""Relleno por parches en Rhino 8 (18-nov): los parches (mallas con vértices propios, sin soldar entre sí) se juntan en
UNA malla y QuadRemesh corre con PreserveMeshArrayEdgesMode (2 = estricto): las orillas entre parches se conservan como
lazos. Después se sueldan los vértices encimados y llevan crease las aristas que caen sobre los cortes.
Variables: PARCHES (carpeta con parche_*.obj y cortes.json), SALIDA, LARGO (8), MODO (2), TOLC (0.3)
"""
import json, os, glob
import Rhino, Rhino.Geometry as rg
g = globals()
D, SALIDA = g['PARCHES'], g['SALIDA']
LARGO = float(g.get('LARGO', 8.0)); MODO = int(g.get('MODO', 2)); TOLC = float(g.get('TOLC', 0.3))
todo = rg.Mesh()
for p in sorted(glob.glob(os.path.join(D, 'parche_*.obj'))):
    V = []; F = []
    for l in open(p):
        if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith('f '): F.append([int(x) - 1 for x in l.split()[1:]])
    me = rg.Mesh()
    for v in V: me.Vertices.Add(*v)
    for f in F:
        if len(f) == 4: me.Faces.AddFace(*f)
        else: me.Faces.AddFace(f[0], f[1], f[2])
    todo.Append(me)
todo.Normals.ComputeNormals(); todo.Compact()
prm = rg.QuadRemeshParameters()
prm.TargetEdgeLength = LARGO; prm.AdaptiveSize = 0; prm.AdaptiveQuadCount = False
prm.DetectHardEdges = False; prm.PreserveMeshArrayEdgesMode = MODO
qm = todo.QuadRemesh(prm)
if qm is None: raise RuntimeError('QuadRemesh devolvió None')
qm.Vertices.CombineIdentical(True, True); qm.Weld(3.14159); qm.Compact()
cortes = json.load(open(os.path.join(D, 'cortes.json')))['cortes']
segs = [rg.Line(rg.Point3d(*a), rg.Point3d(*b)) for a, b in cortes]
def dcorte(p):
    return min(s.DistanceTo(p, True) for s in segs)
QV = [rg.Point3d(qm.Vertices[i]) for i in range(qm.Vertices.Count)]
dv = [dcorte(p) for p in QV]
tv = qm.TopologyVertices
crease = []
for ei in range(qm.TopologyEdges.Count):
    a, b = qm.TopologyEdges.GetTopologyVertices(ei)
    ia, ib = tv.MeshVertexIndices(a)[0], tv.MeshVertexIndices(b)[0]
    if dv[ia] < TOLC and dv[ib] < TOLC and dcorte((QV[ia] + QV[ib]) / 2) < TOLC:
        crease.append([ia, ib])
os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
nt = 0
with open(SALIDA, 'w', encoding='utf-8') as fh:
    fh.write('# relleno por parches, modo %d, largo %.1f\n' % (MODO, LARGO))
    for p in QV: fh.write('v %.6f %.6f %.6f\n' % (p.X, p.Y, p.Z))
    for i in range(qm.Faces.Count):
        f = qm.Faces[i]
        if f.IsQuad: fh.write('f %d %d %d %d\n' % (f.A + 1, f.B + 1, f.C + 1, f.D + 1))
        else: fh.write('f %d %d %d\n' % (f.A + 1, f.B + 1, f.C + 1)); nt += 1
json.dump({'crease': crease}, open(SALIDA[:-4] + '_crease.json', 'w'))
print(json.dumps({'caras': qm.Faces.Count, 'triangulos': nt, 'vertices': qm.Vertices.Count, 'crease': len(crease),
                  'desnudas': sum(1 for e in range(qm.TopologyEdges.Count) if qm.TopologyEdges.GetConnectedFaces(e).Length == 1)}))
