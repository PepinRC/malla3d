"""Pega lazos de la jaula sobre las guías sin zigzag (18-nov, 14:15).
Uso: python pegar_lazos.py <jaula.obj> <guias.json> <zona.obj> <salida.obj> [iter_relajar=6]
Por guía: camino de aristas entre los vértices más cercanos a sus puntas, con costo
largo * (1 + 4*d + 8*(1-|cos|)) (d = distancia del punto medio a la guía, cos = arista contra la tangente de la guía);
aristas a más de 4 mm se prohíben. Los vértices del camino van al punto más cercano de la guía y sus aristas llevan
crease. Los demás se relajan (promedio de vecinos, 0.5) y se reproyectan a la zona; los del borde se reproyectan al
borde de la zona. Reporta cuadros volteados respecto a la zona."""
import os as _os; RAIZ = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # raíz del repositorio
import json, sys, heapq, numpy as np
from scipy.spatial import cKDTree
J, G, Z, OUT = sys.argv[1:5]; IT = int(sys.argv[5]) if len(sys.argv) > 5 else 6; DMAX = float(sys.argv[6]) if len(sys.argv) > 6 else 4.0; APOYO = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] != '-' else None; EXCL = len(sys.argv) > 8 and sys.argv[8] == '1'
def leer(p):
    V, F = [], []
    for l in open(p, encoding='utf-8'):
        if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith('f '): F.append([int(x.split('/')[0]) - 1 for x in l.split()[1:]])
    return np.array(V), F
def densa(P, h):
    out = [P[0]]
    for k in range(len(P) - 1):
        l = np.linalg.norm(P[k + 1] - P[k]); m = max(1, int(l / h))
        for s in range(1, m + 1): out.append(P[k] + (P[k + 1] - P[k]) * s / m)
    return np.array(out)
def muestras(V, F, h):
    out, nor = [], []
    for f in F:
        for k in range(1, len(f) - 1):
            A, B, C = V[f[0]], V[f[k]], V[f[k + 1]]; nn = np.cross(B - A, C - A); nn /= max(np.linalg.norm(nn), 1e-12)
            n = max(2, int(max(np.linalg.norm(B - A), np.linalg.norm(C - A), np.linalg.norm(C - B)) / h) + 1)
            for i in range(n + 1):
                for j in range(n + 1 - i): out.append(A + (B - A) * i / n + (C - A) * j / n); nor.append(nn)
    return np.array(out), np.array(nor)
V, F = leer(J); ZV, ZF = leer(Z)
S, SN = muestras(ZV, ZF, 0.15); TS = cKDTree(S)
ze = {}
for f in ZF:
    for k in range(len(f)):
        e = tuple(sorted((f[k], f[(k + 1) % len(f)]))); ze[e] = ze.get(e, 0) + 1
zb = [densa(ZV[[a, b]], 0.1) for (a, b), c in ze.items() if c == 1]
TB = cKDTree(np.concatenate(zb))
BP = np.concatenate(zb)
adj = {}; ed = {}
for f in F:
    for k in range(len(f)):
        a, b = f[k], f[(k + 1) % len(f)]
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
        e = tuple(sorted((a, b))); ed[e] = ed.get(e, 0) + 1
borde = {v for e, c in ed.items() if c == 1 for v in e}
TV = cKDTree(V); fijos = {}; crease = set(); rep = []
_lista = [(g_, True) for g_ in json.load(open(G, encoding='utf-8'))] + ([(g_, False) for g_ in json.load(open(APOYO, encoding='utf-8'))] if APOYO else [])
if EXCL: _lista.sort(key=lambda x: (not x[1], -sum(np.linalg.norm(np.diff(np.array(x[0]['puntos']), axis=0), axis=1))))
for gu, es_crease in _lista:
    P = densa(np.array(gu['puntos']), 0.25); TG = cKDTree(P)
    tan = np.gradient(P, axis=0); tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12)
    s = int(TV.query(P[0])[1]); t = int(TV.query(P[-1])[1])
    if s == t: rep.append((gu['n'], 'puntas iguales')); continue
    dist = {s: 0.0}; prev = {}; pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == t: break
        if d > dist.get(u, 1e18): continue
        for w in adj.get(u, ()):
            m = (V[u] + V[w]) / 2; dm, im = TG.query(m)
            if dm > DMAX: continue
            if (EXCL or not es_crease) and w in fijos and w != t: continue
            e = V[w] - V[u]; L = np.linalg.norm(e); c = abs(np.dot(e / max(L, 1e-12), tan[im]))
            cost = L * (1 + 4 * dm + 8 * (1 - c))
            if d + cost < dist.get(w, 1e18): dist[w] = d + cost; prev[w] = u; heapq.heappush(pq, (d + cost, w))
    if t not in prev: rep.append((gu['n'], 'sin camino')); continue
    cam = [t]
    while cam[-1] != s: cam.append(prev[cam[-1]])
    # reparto a lo largo de la guía por largo de arco (sin dos vértices en el mismo punto)
    ii = [int(TG.query(V[v])[1]) for v in cam]
    if ii[0] > ii[-1]: cam = cam[::-1]; ii = ii[::-1]
    cum = np.concatenate([[0], np.cumsum([np.linalg.norm(V[b] - V[a]) for a, b in zip(cam[:-1], cam[1:])])])
    cum = cum / max(cum[-1], 1e-12)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    a0, a1 = arc[ii[0]], arc[ii[-1]]
    # si la punta de la guía queda a < 6 mm del vértice de la punta, el lazo llega hasta la punta
    if np.linalg.norm(V[cam[0]] - P[0]) < 6 and cam[0] not in borde: a0 = 0.0
    if np.linalg.norm(V[cam[-1]] - P[-1]) < 6 and cam[-1] not in borde: a1 = arc[-1]
    for v, u in zip(cam, cum):
        q = P[int(np.searchsorted(arc, a0 + u * (a1 - a0)).clip(0, len(P) - 1))]
        if es_crease or v not in fijos: fijos[v] = q if v not in fijos else (fijos[v] + q) / 2
    if es_crease:
        for a, b in zip(cam[:-1], cam[1:]): crease.add(tuple(sorted((a, b))))
    rep.append((gu['n'], len(cam)))
# dos vértices vecinos que caen en el mismo punto (donde se juntan dos guías): se suelta el más lejano
orig = V.copy()
for (a, b) in list(ed):
    if a in fijos and b in fijos and np.linalg.norm(fijos[a] - fijos[b]) < 0.5:
        suelta = a if np.linalg.norm(orig[a] - fijos[a]) > np.linalg.norm(orig[b] - fijos[b]) else b
        del fijos[suelta]
for v, p in fijos.items(): V[v] = p
libres = [v for v in range(len(V)) if v not in fijos and v not in borde]  # el borde de la zona no se mueve
for _ in range(IT):
    W = V.copy()
    for v in libres:
        nb = list(adj.get(v, ()))
        if v in borde: nb = [w for w in nb if w in borde]
        if nb: W[v] = 0.5 * V[v] + 0.5 * V[nb].mean(0)
    V = W
    for v in libres:
        V[v] = BP[TB.query(V[v])[1]] if v in borde else S[TS.query(V[v])[1]]
# cuadros volteados respecto a la zona: se desenredan con suavizado local de sus vértices libres
sys.path.insert(0, RAIZ + "/nucleo/guiado_v5"); sys.dont_write_bytecode = True
from ajustar import j as _j
def volteadas():
    out = []
    if all(len(f) == 4 for f in F):
        nn, _ = _j.normales(V, F); out = [int(i) for i in np.flatnonzero(_j.crossed_faces(V, F, nn))]
    for fi, f in enumerate(F):
        c = V[f].mean(0); n = np.zeros(3)
        for k in range(len(f)): n += np.cross(V[f[k]], V[f[(k + 1) % len(f)]])
        if np.dot(n, SN[TS.query(c)[1]]) < 0: out.append(fi)
    return out
fijo_set = set(fijos)
vol0 = len(volteadas())
for ronda in range(30):
    vf = volteadas()
    if not vf: break
    zona_v = {v for fi in vf for v in F[fi]}
    zona_v |= {w for v in list(zona_v) for w in adj.get(v, ())}
    if ronda >= 15: zona_v |= {w for v in list(zona_v) for w in adj.get(v, ())}
    mov = [v for v in zona_v if v not in fijo_set]
    for _ in range(3):
        W = V.copy()
        for v in mov:
            nb = list(adj.get(v, ()))
            if v in borde: nb = [w for w in nb if w in borde]
            if nb: W[v] = 0.3 * V[v] + 0.7 * V[nb].mean(0)
        V = W
        for v in mov: V[v] = BP[TB.query(V[v])[1]] if v in borde else S[TS.query(V[v])[1]]
volt = len(volteadas())
print('volteados antes de desenredar', vol0, 'despues', volt, file=sys.stderr)
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('# lazos pegados sin zigzag\n')
    for p in V: fh.write('v %.6f %.6f %.6f\n' % tuple(p))
    for f in F: fh.write('f ' + ' '.join(str(i + 1) for i in f) + '\n')
json.dump({'crease': [list(e) for e in sorted(crease)]}, open(OUT[:-4] + '_crease.json', 'w'))
print(json.dumps({'guias': len(rep), 'pegadas': sum(isinstance(x[1], int) for x in rep), 'fallas': [x for x in rep if not isinstance(x[1], int)],
                  'vertices_fijos': len(fijos), 'crease': len(crease), 'volteados': volt}))
