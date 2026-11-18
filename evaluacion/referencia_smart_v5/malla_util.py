"""Utilidades numpy para la referencia v4.1 (solo HD)."""
import hashlib
import numpy as np


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def leer_obj(ruta):
    V, F = [], []
    with open(ruta, "r", encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            if ln.startswith("v "):
                p = ln.split()
                V.append((float(p[1]), float(p[2]), float(p[3])))
            elif ln.startswith("f "):
                idx = [int(t.split("/")[0]) for t in ln.split()[1:]]
                F.append([i - 1 if i > 0 else len(V) + i for i in idx])
    return np.array(V, float), F


def triangular(F):
    """Abanico; devuelve tris y el índice de cara original de cada tri."""
    T, orig = [], []
    for k, f in enumerate(F):
        for j in range(1, len(f) - 1):
            T.append((f[0], f[j], f[j + 1]))
            orig.append(k)
    return np.array(T, int), np.array(orig, int)


def soldar(V, T, tol=1e-3):
    q = np.round(V / tol).astype(np.int64)
    _, inv, = np.unique(q, axis=0, return_inverse=True)[0:2]
    inv = inv.reshape(-1)
    nV = inv.max() + 1
    Vw = np.zeros((nV, 3)); cnt = np.zeros(nV)
    np.add.at(Vw, inv, V); np.add.at(cnt, inv, 1)
    Vw /= cnt[:, None]
    Tw = inv[T]
    return Vw, Tw, inv


def geometria_caras(V, T):
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    cr = np.cross(b - a, c - a)
    nrm = np.linalg.norm(cr, axis=1)
    area = 0.5 * nrm
    n = cr / np.maximum(nrm, 1e-15)[:, None]
    cen = (a + b + c) / 3.0
    return n, area, cen


def aristas(T):
    """Devuelve E (m,2) únicas, ef (lista de caras por arista) y para cada cara sus 3 aristas."""
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    fi = np.concatenate([np.arange(len(T))] * 3)
    es = np.sort(e, axis=1)
    E, inv = np.unique(es, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    cuenta = np.bincount(inv, minlength=len(E))
    ef = [[] for _ in range(len(E))]
    for k in order:
        ef[inv[k]].append(fi[k])
    fe = inv.reshape(3, -1).T
    return E, ef, cuenta, fe


class Rejilla:
    def __init__(self, P, h):
        self.h = h
        self.P = P
        q = np.floor(P / h).astype(np.int64)
        self.q = q
        d = {}
        for i, k in enumerate(map(tuple, q)):
            d.setdefault(k, []).append(i)
        self.d = {k: np.array(v) for k, v in d.items()}

    def cerca(self, p, r):
        lo = np.floor((p - r) / self.h).astype(int)
        hi = np.floor((p + r) / self.h).astype(int)
        res = []
        for i in range(lo[0], hi[0] + 1):
            for j in range(lo[1], hi[1] + 1):
                for k in range(lo[2], hi[2] + 1):
                    v = self.d.get((i, j, k))
                    if v is not None:
                        res.append(v)
        if not res:
            return np.zeros(0, int)
        idx = np.concatenate(res)
        dd = np.linalg.norm(self.P[idx] - p, axis=1)
        return idx[dd <= r]
