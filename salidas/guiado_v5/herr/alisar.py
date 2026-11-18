"""Alisa la jaula sin salirse de la forma: promedio de vecinos en los vértices libres y, si un vértice queda a más de
tol de la zona, se regresa hasta tol. Líneas y borde quietos. Uso: python alisar.py <jaula> <zona> <salida> [tol=0.15] [iter=20] [paso=0.4]"""
import json, sys, shutil, numpy as np
from scipy.spatial import cKDTree
J, Z, OUT = sys.argv[1:4]; TOL = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
IT = int(sys.argv[5]) if len(sys.argv) > 5 else 20; PASO = float(sys.argv[6]) if len(sys.argv) > 6 else 0.4
def leer(p):
    V, F = [], []
    for l in open(p, encoding='utf-8'):
        if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith('f '): F.append([int(x.split('/')[0]) - 1 for x in l.split()[1:]])
    return np.array(V), F
V, F = leer(J); ZV, ZF = leer(Z)
S = []
for f in ZF:
    P = ZV[f]
    for k in range(1, len(f) - 1):
        A, B, C = P[0], P[k], P[k + 1]; m = max(2, int(max(np.linalg.norm(B - A), np.linalg.norm(C - A)) / 0.12) + 1)
        for i in range(m + 1):
            for j in range(m + 1 - i): S.append(A + (B - A) * i / m + (C - A) * j / m)
S = np.array(S); T = cKDTree(S)
cr = json.load(open(J[:-4] + '_crease.json'))['crease']; fijos = {v for e in cr for v in e}
adj = {}; ed = {}
for f in F:
    for k in range(len(f)):
        a, b = f[k], f[(k + 1) % len(f)]; adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
        e = tuple(sorted((a, b))); ed[e] = ed.get(e, 0) + 1
borde = {v for e, c in ed.items() if c == 1 for v in e}
lib = np.array([v for v in range(len(V)) if v not in fijos and v not in borde])
nb = [list(adj[v]) for v in lib]
for _ in range(IT):
    M = np.array([V[n].mean(0) for n in nb]); V[lib] = V[lib] + PASO * (M - V[lib])
    d, i = T.query(V[lib]); c = S[i]; far = d > TOL
    V[lib[far]] = c[far] + (V[lib[far]] - c[far]) * (TOL / d[far])[:, None]
with open(OUT, 'w', encoding='utf-8') as fh:
    for p in V: fh.write('v %.6f %.6f %.6f\n' % tuple(p))
    for f in F: fh.write('f ' + ' '.join(str(i + 1) for i in f) + '\n')
shutil.copy(J[:-4] + '_crease.json', OUT[:-4] + '_crease.json')
print('alisado', len(lib), 'vertices')
