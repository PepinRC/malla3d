"""Corta la zona de la Smart en parches por las guías (18-nov, 12:40).
Uso: python cortar_parches.py <zona.obj> <guias.json> <salida_dir> [tol_mm=1.2]
1. Cada guía (línea media de un bisel fundido) se asocia a las aristas de la Smart que corren a < tol de ella
   y cuyo diedro es >= 20° (son orillas de bisel u otros dobleces). De cada bisel queda la orilla con más largo cerca.
2. Esas aristas son cortes. Los parches son los componentes de caras separados por cortes.
3. Escribe <salida_dir>/parche_NN.obj (con coordenadas exactas compartidas) y cortes.json (aristas de corte con coords).
"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict

Z, G, OUT = sys.argv[1:4]
TOL = float(sys.argv[4]) if len(sys.argv) > 4 else 1.2
os.makedirs(OUT, exist_ok=True)
V, F = [], []
for l in open(Z, encoding="utf-8"):
    if l.startswith("v "): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith("f "): F.append([int(x) - 1 for x in l.split()[1:]])
V = np.array(V)


def normal(f):
    n = np.zeros(3)
    for k in range(len(f)):
        n += np.cross(V[f[k]], V[f[(k + 1) % len(f)]])
    return n / max(np.linalg.norm(n), 1e-12)


N = np.array([normal(f) for f in F])
ed = defaultdict(list)
for fi, f in enumerate(F):
    for k in range(len(f)):
        ed[tuple(sorted((f[k], f[(k + 1) % len(f)])))].append(fi)
diedro = {}
for e, fs in ed.items():
    if len(fs) == 2:
        c = np.clip(np.dot(N[fs[0]], N[fs[1]]), -1, 1)
        diedro[e] = np.degrees(np.arccos(c))


def densa(P, h=0.2):
    out = [P[0]]
    for k in range(len(P) - 1):
        l = np.linalg.norm(P[k + 1] - P[k]); m = max(1, int(l / h))
        for s in range(1, m + 1): out.append(P[k] + (P[k + 1] - P[k]) * s / m)
    return np.array(out)


cortes = set()
por_guia = []
for gu in json.load(open(G, encoding="utf-8")):
    P = densa(np.array(gu["puntos"])); T = cKDTree(P)
    cand = []
    for e, d in diedro.items():
        if d < 20: continue
        a, b = V[e[0]], V[e[1]]
        da, db, dm = T.query(a)[0], T.query(b)[0], T.query((a + b) / 2)[0]
        if max(da, db, dm) < TOL:
            cand.append((e, np.linalg.norm(b - a), (a + b) / 2))
    if not cand:
        por_guia.append({"n": gu["n"], "aristas": 0}); continue
    # dos orillas de un bisel: se separan por el lado de la guía; se queda el lado con más largo
    lados = defaultdict(list)
    for e, l, m in cand:
        i = T.query(m)[1]; j = min(i + 1, len(P) - 1); i0 = max(i - 1, 0)
        t = P[j] - P[i0]; t /= max(np.linalg.norm(t), 1e-12)
        nf = N[ed[e][0]] + N[ed[e][1]]; nf /= max(np.linalg.norm(nf), 1e-12)
        lado = np.sign(np.dot(np.cross(t, nf), m - P[i]))
        lados[lado].append((e, l))
    mejor = max(lados.values(), key=lambda xs: sum(l for _, l in xs))
    for e, _ in mejor:
        cortes.add(e)
    por_guia.append({"n": gu["n"], "aristas": len(mejor), "largo_mm": round(sum(l for _, l in mejor), 1)})
# cerrar: desde cada punta suelta se sigue por la arista más recta y más marcada hasta otro corte o el borde
borde_v = {v for e, fs in ed.items() if len(fs) == 1 for v in e}
vec = defaultdict(set)
for (a, b) in ed: vec[a].add(b); vec[b].add(a)
def grado(v): return sum(1 for w in vec[v] if tuple(sorted((v, w))) in cortes)
cerradas = 0
for _ in range(3):
    puntas = [v for v in {x for e in cortes for x in e} if grado(v) == 1 and v not in borde_v]
    for v0 in puntas:
        if grado(v0) != 1: continue
        w0 = [w for w in vec[v0] if tuple(sorted((v0, w))) in cortes][0]
        d = V[v0] - V[w0]; d /= max(np.linalg.norm(d), 1e-12)
        cam = [v0]; largo = 0.0; ok = False
        while largo < 80:
            u = cam[-1]; mejor = None
            for w in vec[u]:
                e = tuple(sorted((u, w)))
                if w in cam or e in cortes: continue
                t = V[w] - V[u]; l = np.linalg.norm(t); t /= max(l, 1e-12)
                s = np.dot(t, d) + 0.02 * diedro.get(e, 0)
                if np.dot(t, d) > 0.3 and (mejor is None or s > mejor[0]): mejor = (s, w, t, l)
            if mejor is None: break
            _, w, t, l = mejor
            cam.append(w); largo += l; d = 0.6 * d + 0.4 * t; d /= np.linalg.norm(d)
            if w in borde_v or grado(w) > 0: ok = True; break
        if ok:
            for a, b in zip(cam[:-1], cam[1:]): cortes.add(tuple(sorted((a, b))))
            cerradas += 1
print("puntas cerradas", cerradas)
# parches: componentes de caras sin cruzar cortes
padre = list(range(len(F)))
def raiz(x):
    while padre[x] != x:
        padre[x] = padre[padre[x]]; x = padre[x]
    return x
for e, fs in ed.items():
    if len(fs) == 2 and e not in cortes:
        a, b = raiz(fs[0]), raiz(fs[1])
        if a != b: padre[a] = b
grupos = defaultdict(list)
for fi in range(len(F)): grupos[raiz(fi)].append(fi)
grupos = sorted(grupos.values(), key=len, reverse=True)
for k, fs in enumerate(grupos):
    us = sorted({v for fi in fs for v in F[fi]}); m = {v: i for i, v in enumerate(us)}
    with open(os.path.join(OUT, "parche_%02d.obj" % k), "w", encoding="utf-8") as fh:
        for v in us: fh.write("v %.6f %.6f %.6f\n" % tuple(V[v]))
        for fi in fs: fh.write("f " + " ".join(str(m[v] + 1) for v in F[fi]) + "\n")
json.dump({"cortes": [[V[a].tolist(), V[b].tolist()] for a, b in sorted(cortes)], "por_guia": por_guia,
           "parches": [len(g) for g in grupos]}, open(os.path.join(OUT, "cortes.json"), "w"), indent=1)
print("cortes", len(cortes), "parches", len(grupos), "caras por parche", [len(g) for g in grupos])
print("guias sin aristas:", [p["n"] for p in por_guia if not p["aristas"]])
