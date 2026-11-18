"""Medición rápida de una jaula de zona (18-nov). Uso: python medir_zona.py <jaula.obj> [zona.obj] [guias.json]
Calidad (verde/amarillo/rojo, criterio de mapa_calidad.py), triángulos, polos interiores 6+ y 3/5,
C5 de la JAULA de control (no de la SubD límite) contra la zona en los dos sentidos (muestras por área),
C8 aproximado: % del largo de las guías a < 0.5 mm de una arista con crease de la jaula.
"""
import json, sys, os
import numpy as np
from scipy.spatial import cKDTree

AQUI = os.path.dirname(os.path.abspath(__file__))
J = sys.argv[1]
Z = sys.argv[2] if len(sys.argv) > 2 else os.path.join(AQUI, "zona_frente.obj")
G = sys.argv[3] if len(sys.argv) > 3 else os.path.join(AQUI, "guias_frente_v0.json")


def leer(p):
    V, F = [], []
    for l in open(p, encoding="utf-8"):
        if l.startswith("v "): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("f "): F.append([int(x.split("/")[0]) - 1 for x in l.split()[1:]])
    return np.array(V), F


def muestras(V, F, h=0.8):
    out = []
    for f in F:
        tris = [(f[0], f[k], f[k + 1]) for k in range(1, len(f) - 1)]
        for a, b, c in tris:
            A, B, C = V[a], V[b], V[c]
            ar = 0.5 * np.linalg.norm(np.cross(B - A, C - A))
            n = max(1, int(ar / (h * h)))
            r = np.random.default_rng(len(out)).random((n, 2))
            m = r.sum(1) > 1; r[m] = 1 - r[m]
            out.append(A + r[:, :1] * (B - A) + r[:, 1:] * (C - A))
    return np.concatenate(out)


def dist_a_malla(P, V, F, Q):
    # distancia aproximada: a muestras densas de la otra malla
    return cKDTree(Q).query(P)[0]


V, F = leer(J)
ZV, ZF = leer(Z)
ed = {}
for f in F:
    for k in range(len(f)):
        e = tuple(sorted((f[k], f[(k + 1) % len(f)]))); ed[e] = ed.get(e, 0) + 1
val = np.zeros(len(V), int)
for (i, j) in ed: val[i] += 1; val[j] += 1
borde = {v for e, c in ed.items() if c == 1 for v in e}
interior = [i for i in range(len(V)) if i not in borde and val[i] > 0]
n = [0, 0, 0]
for f in F:
    P = V[f]; L = np.linalg.norm(P - np.roll(P, -1, 0), axis=1); r = L.max() / max(L.min(), 1e-9)
    if r <= 2 and all(val[i] == 4 or i in borde for i in f): n[0] += 1
    elif r <= 3.5: n[1] += 1
    else: n[2] += 1
t = len(F)
Sj, Sz = muestras(V, F, 0.4), muestras(ZV, ZF, 0.4)
d1 = dist_a_malla(muestras(V, F), V, F, Sz); d2 = dist_a_malla(muestras(ZV, ZF), ZV, ZF, Sj)
res = {"caras": t, "triangulos": sum(len(f) != 4 for f in F),
       "verde_pct": round(100 * n[0] / t, 1), "amarillo_pct": round(100 * n[1] / t, 1), "rojo_pct": round(100 * n[2] / t, 1),
       "polos6_interiores": int(sum(val[i] >= 6 for i in interior)),
       "polos3o5_interiores": int(sum(val[i] in (3, 5) for i in interior)),
       "C5_jaula_a_zona_media_p95": [round(float(d1.mean()), 3), round(float(np.percentile(d1, 95)), 3)],
       "C5_zona_a_jaula_media_p95": [round(float(d2.mean()), 3), round(float(np.percentile(d2, 95)), 3)]}
cj = J[:-4] + "_crease.json"
if os.path.exists(cj) and os.path.exists(G):
    cr = json.load(open(cj))["crease"]
    seg = []
    for a, b in cr:
        A, B = V[a], V[b]
        for s in np.linspace(0, 1, 9): seg.append(A + s * (B - A))
    tot = ok = 0.0
    if seg:
        T = cKDTree(np.array(seg))
        for gu in json.load(open(G, encoding="utf-8")):
            P = np.array(gu["puntos"])
            for k in range(len(P) - 1):
                l = np.linalg.norm(P[k + 1] - P[k])
                for s in np.linspace(0, 1, max(2, int(l / 0.5))):
                    q = P[k] + s * (P[k + 1] - P[k]); tot += 1
                    ok += T.query(q)[0] < 0.5
    res["C8_aprox_cobertura_pct"] = round(100 * ok / max(tot, 1), 1)
print(json.dumps(res, ensure_ascii=False))
