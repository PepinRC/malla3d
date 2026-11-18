"""Lazos de apoyo: a cada lado de cada línea del diseño, una línea paralela sobre la superficie a d mm (como los
lazos de apoyo de un modelador). Se recortan donde se acercan a otra línea o al borde. Uso:
python lineas_apoyo.py <guias.json> <zona.obj> <salida.json> [d=3] [largo_min=12]"""
import json, sys, numpy as np
from scipy.spatial import cKDTree
G, Z, OUT = sys.argv[1:4]; D = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0; LMIN = float(sys.argv[5]) if len(sys.argv) > 5 else 12.0
V, F = [], []
for l in open(Z, encoding='utf-8'):
    if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith('f '): F.append([int(x) - 1 for x in l.split()[1:]])
V = np.array(V); S = []; SN = []; ed = {}
for f in F:
    P = V[f]; n = np.zeros(3)
    for k in range(len(f)): n += np.cross(P[k], P[(k + 1) % len(f)])
    n /= max(np.linalg.norm(n), 1e-12)
    for k in range(len(f)): e = tuple(sorted((f[k], f[(k + 1) % len(f)]))); ed[e] = ed.get(e, 0) + 1
    for k in range(1, len(f) - 1):
        A, B, C = P[0], P[k], P[k + 1]; m = max(2, int(max(np.linalg.norm(B - A), np.linalg.norm(C - A)) / 0.2) + 1)
        for i in range(m + 1):
            for j in range(m + 1 - i): S.append(A + (B - A) * i / m + (C - A) * j / m); SN.append(n)
S = np.array(S); SN = np.array(SN); TS = cKDTree(S)
B = []
for (a, b), c in ed.items():
    if c == 1:
        for s in np.linspace(0, 1, 30): B.append(V[a] + s * (V[b] - V[a]))
TB = cKDTree(np.array(B))
def dens(P, h=0.5):
    P = np.array(P); out = [P[0]]
    for k in range(len(P) - 1):
        m = max(1, int(np.linalg.norm(P[k + 1] - P[k]) / h))
        for s in range(1, m + 1): out.append(P[k] + (P[k + 1] - P[k]) * s / m)
    return np.array(out)
gs = json.load(open(G, encoding='utf-8')); D_ = [dens(g['puntos']) for g in gs]
TG = cKDTree(np.concatenate(D_))
out = []
for g, P in zip(gs, D_):
    t = np.gradient(P, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    for lado in (1, -1):
        Q = []
        for p, tt in zip(P, t):
            # normal del lado: se mira la cara a d/2 de cada lado
            n = SN[TS.query(p)[1]]; b = np.cross(n, tt); b /= max(np.linalg.norm(b), 1e-12)
            q = p + lado * 0.5 * D * b; nq = SN[TS.query(q)[1]]; bq = np.cross(nq, tt); bq /= max(np.linalg.norm(bq), 1e-12)
            q = S[TS.query(p + lado * D * (b + bq) / max(np.linalg.norm(b + bq), 1e-12))[1]]
            Q.append(q)
        Q = np.array(Q)
        ok = (TG.query(Q)[0] > 0.7 * D) & (TB.query(Q)[0] > 2.0)
        i = 0
        while i < len(Q):
            if ok[i]:
                j = i
                while j < len(Q) and ok[j]: j += 1
                R = Q[i:j]
                if len(R) > 1 and np.linalg.norm(np.diff(R, axis=0), axis=1).sum() >= LMIN:
                    k = list(range(0, len(R), 4)) + [len(R) - 1]
                    out.append({'n': g['n'] * 10 + (1 if lado > 0 else 2), 'apoyo_de': g['n'], 'puntos': R[k].tolist()})
                i = j
            else: i += 1
json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1)
print(len(out), 'lineas de apoyo; largo', round(sum(np.linalg.norm(np.diff(np.array(o['puntos']), axis=0), axis=1).sum() for o in out)), 'mm')
