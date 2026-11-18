"""Líneas principales v2 (18-nov, 14:45): de cada haz de líneas paralelas se queda la MÁS LARGA y, de las demás,
solo los tramos que se alejan más de sep de lo ya elegido (así no se pierde ningún doblez). Se recorta a la zona,
se quitan tramos cortos y las puntas sueltas a < 8 mm de otra línea se alargan hasta tocarla.
Uso: python guias_principales2.py <candidatas.json> <zona.obj> <salida.json> [sep=6] [largo_min=20]"""
import json, sys, numpy as np
from scipy.spatial import cKDTree
C, Z, OUT = sys.argv[1:4]
SEP = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0; LMIN = float(sys.argv[5]) if len(sys.argv) > 5 else 20.0; EXT = len(sys.argv) > 6 and sys.argv[6] == "1"
def dens(P, h=0.5):
    P = np.array(P); out = [P[0]]
    for k in range(len(P) - 1):
        m = max(1, int(np.linalg.norm(P[k + 1] - P[k]) / h))
        for s in range(1, m + 1): out.append(P[k] + (P[k + 1] - P[k]) * s / m)
    return np.array(out)
V, F = [], []
for l in open(Z, encoding='utf-8'):
    if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith('f '): F.append([int(x) - 1 for x in l.split()[1:]])
V = np.array(V); S = []
for f in F:
    for k in range(1, len(f) - 1):
        A, B, Cc = V[f[0]], V[f[k]], V[f[k + 1]]; n = max(2, int(max(np.linalg.norm(B - A), np.linalg.norm(Cc - A)) / 0.5) + 1)
        for i in range(n + 1):
            for j in range(n + 1 - i): S.append(A + (B - A) * i / n + (Cc - A) * j / n)
TS = cKDTree(np.array(S))
def tramos(d, mask):
    out = []; i = 0
    while i < len(d):
        if mask[i]:
            j = i
            while j < len(d) and mask[j]: j += 1
            if (j - i) * 0.5 >= LMIN: out.append(d[i:j])
            i = j
        else: i += 1
    return out
cands = sorted(json.load(open(C, encoding='utf-8')), key=lambda c: -c['largo_mm'])
elegidos = []  # (n, puntos densos)
for c in cands:
    d = dens(c['puntos'])
    m = TS.query(d)[0] < 1.5
    if elegidos:
        T = cKDTree(np.concatenate([e[1] for e in elegidos])); m &= T.query(d)[0] > SEP
    for t in tramos(d, m): elegidos.append((c['n'], t))
# puntas sueltas: alargar hasta la línea más cercana si está a < 8 mm
res = []
for i, (n, d) in enumerate(elegidos):
    otros = [e[1] for k, e in enumerate(elegidos) if k != i]
    if otros:
        T = cKDTree(np.concatenate(otros)); P = np.concatenate(otros)
        for lado in (0, -1):
            dist, ix = T.query(d[lado])
            if EXT and 0.5 < dist < 8:
                ext = dens([d[lado], P[ix]])
                d = np.vstack([ext[::-1][:-1], d]) if lado == 0 else np.vstack([d, ext[1:]])
    k = [0]; acc = 0
    for m_ in range(1, len(d)):
        acc += np.linalg.norm(d[m_] - d[m_ - 1])
        if acc >= 2: k.append(m_); acc = 0
    if k[-1] != len(d) - 1: k.append(len(d) - 1)
    res.append({'n': n, 'puntos': d[k].tolist(), 'largo_mm': round(float(np.linalg.norm(np.diff(d, axis=0), axis=1).sum()), 1)})
json.dump(res, open(OUT, 'w', encoding='utf-8'), indent=1)
print(len(res), 'lineas; largo total', round(sum(r['largo_mm'] for r in res)), 'mm; por candidata', sorted({r['n'] for r in res}))
