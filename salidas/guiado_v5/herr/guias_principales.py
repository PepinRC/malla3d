"""Junta las guías paralelas (haces de un mismo bisel o doblez) en una sola, como lo haría una persona.
Uso: python guias_principales.py <guias.json> <salida.json> [sep_mm=6] [frac=0.6] [largo_min=35]
Dos guías son del mismo haz si >= frac de la más corta queda a < sep de la otra. De cada haz se queda la guía
central (la de menor distancia media a las demás); las más cortas que largo_min se quitan si no tocan otra guía."""
import json, sys, numpy as np
from scipy.spatial import cKDTree
G, OUT = sys.argv[1:3]
SEP = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
FR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
LMIN = float(sys.argv[5]) if len(sys.argv) > 5 else 35.0
gs = json.load(open(G, encoding='utf-8'))
def densa(P, h=0.5):
    out = [P[0]]
    for k in range(len(P) - 1):
        l = np.linalg.norm(P[k + 1] - P[k]); m = max(1, int(l / h))
        for s in range(1, m + 1): out.append(P[k] + (P[k + 1] - P[k]) * s / m)
    return np.array(out)
D = [densa(np.array(g['puntos'])) for g in gs]; T = [cKDTree(d) for d in D]
L = [np.linalg.norm(np.diff(np.array(g['puntos']), axis=0), axis=1).sum() for g in gs]
n = len(gs); padre = list(range(n))
def r(x):
    while padre[x] != x: padre[x] = padre[padre[x]]; x = padre[x]
    return x
M = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j: M[i, j] = T[j].query(D[i])[0].mean()
for i in range(n):
    for j in range(i + 1, n):
        a, b = (i, j) if L[i] <= L[j] else (j, i)
        f = (T[b].query(D[a])[0] < SEP).mean()
        if f >= FR: padre[r(a)] = r(b)
haces = {}
for i in range(n): haces.setdefault(r(i), []).append(i)
quedan = []
for h in haces.values():
    if len(h) == 1: c = h[0]
    else: c = min(h, key=lambda i: np.mean([M[i, j] for j in h if j != i]) - 0.001 * L[i])
    quedan.append((c, h))
final = []
for c, h in quedan:
    if L[c] < LMIN:
        toca = any(T[o].query(D[c][[0, -1]])[0].min() < 2.0 for o, _ in quedan if o != c)
        if not toca: continue
    g = dict(gs[c]); g['haz'] = [gs[i]['n'] for i in h]; final.append(g)
json.dump(final, open(OUT, 'w', encoding='utf-8'), indent=1)
print(len(gs), '->', len(final), 'guias;', 'haces:', [g['haz'] for g in final if len(g['haz']) > 1])
