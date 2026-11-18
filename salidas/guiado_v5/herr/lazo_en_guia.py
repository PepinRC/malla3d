"""% del largo de las guías que tiene una arista de la jaula (cualquiera) que corre a lo largo, a < tol mm.
Uso: python lazo_en_guia.py <jaula.obj> [guias.json] [tol=0.5,1.0]"""
import json, sys, os, numpy as np
from scipy.spatial import cKDTree
J = sys.argv[1]; G = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), '..', 'guias_frente_v0.json')
V, F = [], []
for l in open(J, encoding='utf-8'):
    if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith('f '): F.append([int(x.split('/')[0]) - 1 for x in l.split()[1:]])
V = np.array(V); E = set()
for f in F:
    for k in range(len(f)): E.add(tuple(sorted((f[k], f[(k + 1) % len(f)]))))
P, T = [], []
for a, b in E:
    d = V[b] - V[a]; L = np.linalg.norm(d)
    for s in np.linspace(0, 1, max(2, int(L / 0.25))): P.append(V[a] + s * d); T.append(d / max(L, 1e-9))
P = np.array(P); T = np.array(T); K = cKDTree(P)
res = {}
for tol in (0.5, 1.0):
    tot = ok = 0
    for gu in json.load(open(G, encoding='utf-8')):
        Q = np.array(gu['puntos'])
        for k in range(len(Q) - 1):
            t = Q[k + 1] - Q[k]; l = np.linalg.norm(t); t /= max(l, 1e-9)
            for s in np.linspace(0, 1, max(2, int(l / 0.5))):
                q = Q[k] + s * (Q[k + 1] - Q[k]); tot += 1
                ii = K.query_ball_point(q, tol)
                ok += any(abs(np.dot(T[i], t)) > 0.9 for i in ii)
    res['lazo_%.1fmm_pct' % tol] = round(100 * ok / tot, 1)
print(json.dumps(res))
