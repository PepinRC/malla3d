#! python3
"""Relleno guiado en Rhino 8 (18-nov): zona -> QuadRemesh con las guías como curvas de lazo -> OBJ + crease en las guías.
Variables en exec: ZONA, GUIAS, SALIDA, LARGO (mm por arista, def 8), INFLUENCIA (0 aprox, 1 ..., 2 lazo exacto; def 2)
"""
import json, os
import Rhino, Rhino.Geometry as rg
import System
g = globals()
ZONA, GUIAS, SALIDA = g['ZONA'], g['GUIAS'], g['SALIDA']
LARGO = float(g.get('LARGO', 8.0)); TOLC = float(g.get('TOL_CREASE', 0.4)); INFL = int(g.get('INFLUENCIA', 2))
V, F = [], []
for l in open(ZONA, encoding='utf-8'):
    if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith('f '): F.append([int(x) - 1 for x in l.split()[1:]])
me = rg.Mesh()
for p in V: me.Vertices.Add(*p)
for f in F:
    if len(f) == 4: me.Faces.AddFace(f[0], f[1], f[2], f[3])
    else: me.Faces.AddFace(f[0], f[1], f[2])
me.Normals.ComputeNormals(); me.Compact()
guias = json.load(open(GUIAS, encoding='utf-8'))
curvas = []
for gu in guias:
    pts = [rg.Point3d(*p) for p in gu['puntos']]
    pl = rg.Polyline(pts)
    c = pl.ToPolylineCurve()
    # proyectar al plano de la malla: pull a la malla
    if g.get('JALAR', True) and hasattr(c, 'PullToMesh'):
        pulled = c.PullToMesh(me, 0.05)
        curvas.append(pulled if pulled else c)
    else:
        curvas.append(c)
prm = rg.QuadRemeshParameters()
prm.TargetEdgeLength = LARGO
prm.AdaptiveSize = 0
prm.AdaptiveQuadCount = False
prm.DetectHardEdges = bool(g.get('BORDES_DUROS', False))
prm.GuideCurveInfluence = INFL
prm.PreserveMeshArrayEdgesMode = 0
lista = System.Collections.Generic.List[rg.Curve]()
for c in curvas: lista.Add(c)
qm = me.QuadRemesh(prm, lista)
if qm is None:
    raise RuntimeError('QuadRemesh devolvió None')
# crease: aristas cuyos dos extremos quedan a < 0.4 mm de alguna guía
def cerca(p):
    best = 1e9
    for c in curvas:
        ok, t = c.ClosestPoint(p)
        if ok:
            d = p.DistanceTo(c.PointAt(t))
            if d < best: best = d
    return best
QV = [qm.Vertices[i] for i in range(qm.Vertices.Count)]
dist = [cerca(rg.Point3d(v.X, v.Y, v.Z)) for v in QV]
crease = []
tv = qm.TopologyVertices
for ei in range(qm.TopologyEdges.Count):
    a, b = qm.TopologyEdges.GetTopologyVertices(ei)
    ia, ib = tv.MeshVertexIndices(a)[0], tv.MeshVertexIndices(b)[0]
    if dist[ia] < TOLC and dist[ib] < TOLC:
        pm = (rg.Point3d(QV[ia]) + rg.Point3d(QV[ib])) / 2
        if cerca(pm) < TOLC:
            crease.append([ia, ib])
os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
with open(SALIDA, 'w', encoding='utf-8') as fh:
    fh.write('# relleno guiado QuadRemesh, largo %.1f mm, influencia %d, %d guias\n' % (LARGO, INFL, len(curvas)))
    for v in QV: fh.write('v %.6f %.6f %.6f\n' % (v.X, v.Y, v.Z))
    nt = 0
    for i in range(qm.Faces.Count):
        f = qm.Faces[i]
        if f.IsQuad: fh.write('f %d %d %d %d\n' % (f.A + 1, f.B + 1, f.C + 1, f.D + 1))
        else: fh.write('f %d %d %d\n' % (f.A + 1, f.B + 1, f.C + 1)); nt += 1
json.dump({'crease': crease}, open(SALIDA[:-4] + '_crease.json', 'w'))
print(json.dumps({'caras': qm.Faces.Count, 'triangulos': nt, 'vertices': qm.Vertices.Count, 'crease': len(crease)}))
