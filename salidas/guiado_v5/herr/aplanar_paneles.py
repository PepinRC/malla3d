"""Aplana los paneles: cada vértice libre cuya vecindad cae sobre caras de Tripo casi coplanares (< ang°) se lleva
exactamente a ese plano; así la SubD de un panel plano sale plana. Los vértices de las líneas no se mueven.
Uso: python aplanar_paneles.py <jaula.obj> <zona.obj> <salida.obj> [ang=3] [pasadas=3]"""
import json, sys, shutil, numpy as np
from scipy.spatial import cKDTree
J, Z, OUT = sys.argv[1:4]; ANG = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0; NP = int(sys.argv[5]) if len(sys.argv) > 5 else 3
def leer(p):
    V, F = [], []
    for l in open(p, encoding='utf-8'):
        if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith('f '): F.append([int(x.split('/')[0]) - 1 for x in l.split()[1:]])
    return np.array(V), F
V, F = leer(J); ZV, ZF = leer(Z)
cr = json.load(open(J[:-4] + '_crease.json'))['crease']; fijos = {v for e in cr for v in e}
# plano de cada cara de la zona y muestras con su cara
Pn, Pd, S, SI = [], [], [], []
for fi, f in enumerate(ZF):
    P = ZV[f]; n = np.zeros(3)
    for k in range(len(f)): n += np.cross(P[k], P[(k + 1) % len(f)])
    n /= max(np.linalg.norm(n), 1e-12); Pn.append(n); Pd.append(np.dot(n, P.mean(0)))
    for k in range(1, len(f) - 1):
        A, B, C = P[0], P[k], P[k + 1]; m = max(2, int(max(np.linalg.norm(B - A), np.linalg.norm(C - A)) / 0.3) + 1)
        for i in range(m + 1):
            for j in range(m + 1 - i): S.append(A + (B - A) * i / m + (C - A) * j / m); SI.append(fi)
Pn = np.array(Pn); Pd = np.array(Pd); T = cKDTree(np.array(S)); SI = np.array(SI)
adj = {}; ed = {}
for f in F:
    for k in range(len(f)):
        a, b = f[k], f[(k + 1) % len(f)]; adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
        e = tuple(sorted((a, b))); ed[e] = ed.get(e, 0) + 1
borde = {v for e, c in ed.items() if c == 1 for v in e}
cosmin = np.cos(np.radians(ANG)); movidos = 0
for _ in range(NP):
    cara = SI[T.query(V)[1]]
    for v in range(len(V)):
        if v in fijos or v in borde: continue
        anillo = [v] + list(adj.get(v, ()))
        n0 = Pn[cara[v]]
        if all(np.dot(Pn[cara[w]], n0) > cosmin for w in anillo):
            # plano promedio de las caras del anillo, pesado
            n = Pn[cara[anillo]].mean(0); n /= np.linalg.norm(n); d = np.mean([np.dot(n, ZV[ZF[cara[w]]].mean(0)) * 0 + Pd[cara[w]] * np.dot(n, Pn[cara[w]]) for w in anillo])
            V[v] = V[v] - (np.dot(n, V[v]) - d) * n; movidos += 1
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('# paneles aplanados\n')
    for p in V: fh.write('v %.6f %.6f %.6f\n' % tuple(p))
    for f in F: fh.write('f ' + ' '.join(str(i + 1) for i in f) + '\n')
shutil.copy(J[:-4] + '_crease.json', OUT[:-4] + '_crease.json')
print(json.dumps({'movidos_por_pasada': movidos // NP, 'libres': len(V) - len(fijos) - len(borde)}))
