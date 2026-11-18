"""Pega los lazos de la jaula sobre las guías (18-nov, 12:30).
Uso: python pegar_guias.py <jaula.obj> <guias.json> <zona.obj> <salida.obj>
1. Por cada guía: vértices de la jaula más cercanos a sus extremos; camino más corto en la jaula con costo
   largo * (1 + 20 * distancia del punto medio de la arista a la guía). Si el camino se aleja más de 3 mm, se descarta.
2. Los vértices del camino se mueven al punto más cercano de la guía; sus aristas llevan crease.
3. Los vértices que no están en ninguna guía se reproyectan a la zona (muestras densas de la Smart, 0.15 mm)
   después de 3 pasadas de suavizado tangencial suave (0.3), para repartir lo que movió el pegado.
"""
import json, sys, heapq
import numpy as np
from scipy.spatial import cKDTree

J, G, Z, OUT = sys.argv[1:5]


def leer(p):
    V, F = [], []
    for l in open(p, encoding="utf-8"):
        if l.startswith("v "): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("f "): F.append([int(x.split("/")[0]) - 1 for x in l.split()[1:]])
    return np.array(V), F


def muestras(V, F, h):
    out = []
    for f in F:
        for k in range(1, len(f) - 1):
            A, B, C = V[f[0]], V[f[k]], V[f[k + 1]]
            n = max(2, int(max(np.linalg.norm(B - A), np.linalg.norm(C - A), np.linalg.norm(C - B)) / h) + 1)
            for i in range(n + 1):
                for j in range(n + 1 - i):
                    out.append(A + (B - A) * i / n + (C - A) * j / n)
    return np.array(out)


def densa(P, h=0.1):
    out = [P[0]]
    for k in range(len(P) - 1):
        l = np.linalg.norm(P[k + 1] - P[k]); m = max(1, int(l / h))
        for s in range(1, m + 1): out.append(P[k] + (P[k + 1] - P[k]) * s / m)
    return np.array(out)


V, F = leer(J)
ZV, ZF = leer(Z)
S = muestras(ZV, ZF, 0.15); TS = cKDTree(S)
adj = {}
for f in F:
    for k in range(len(f)):
        a, b = f[k], f[(k + 1) % len(f)]
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
TV = cKDTree(V)
fijos = {}
crease = set()
usadas = 0
for gu in json.load(open(G, encoding="utf-8")):
    P = densa(np.array(gu["puntos"]))
    TG = cKDTree(P)
    s = int(TV.query(P[0])[1]); t = int(TV.query(P[-1])[1])
    if s == t:
        continue
    dist = {s: 0.0}; prev = {}; pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == t: break
        if d > dist.get(u, 1e18): continue
        for w in adj.get(u, ()):
            m = (V[u] + V[w]) / 2; dm = TG.query(m)[0]
            c = np.linalg.norm(V[w] - V[u]) * (1 + 20 * dm)
            if d + c < dist.get(w, 1e18):
                dist[w] = d + c; prev[w] = u; heapq.heappush(pq, (d + c, w))
    if t not in prev:
        continue
    cam = [t]
    while cam[-1] != s: cam.append(prev[cam[-1]])
    dmax = max(TG.query(V[v])[0] for v in cam)
    if dmax > 3.0:
        continue
    usadas += 1
    for v in cam:
        if v not in fijos:
            fijos[v] = P[TG.query(V[v])[1]]
    for a, b in zip(cam[:-1], cam[1:]):
        crease.add(tuple(sorted((a, b))))
for v, p in fijos.items():
    V[v] = p
libres = [v for v in range(len(V)) if v not in fijos]
borde = set()
ed = {}
for f in F:
    for k in range(len(f)):
        e = tuple(sorted((f[k], f[(k + 1) % len(f)]))); ed[e] = ed.get(e, 0) + 1
borde = {v for e, c in ed.items() if c == 1 for v in e}
for _ in range(3):
    W = V.copy()
    for v in libres:
        if v in borde: continue
        nb = list(adj.get(v, ()))
        if nb:
            W[v] = 0.7 * V[v] + 0.3 * V[nb].mean(0)
    V = W
    for v in libres:
        V[v] = S[TS.query(V[v])[1]]
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("# jaula con lazos pegados a %d guias\n" % usadas)
    for p in V: fh.write("v %.6f %.6f %.6f\n" % tuple(p))
    for f in F: fh.write("f " + " ".join(str(i + 1) for i in f) + "\n")
json.dump({"crease": [list(e) for e in sorted(crease)]}, open(OUT[:-4] + "_crease.json", "w"))
print(json.dumps({"guias_pegadas": usadas, "vertices_fijos": len(fijos), "crease": len(crease)}))
