"""Parte la Malla Smart a8bc0f28 (ya en mm) en piezas, SIN usar el HD.

Uso: python datos/smart_v5/segmentar.py
Lee SOLO datos/lowpoly/a8bc0f28/cuerpo.obj. Escribe datos/smart_v5/piezas/<id>.obj y datos/smart_v5/piezas.json.

1. Suelda vertices a 1e-3 mm (Tripo repite vertices en costuras de UV) y saca componentes conexos por arista.
2. Por componente: caras, % triangulos, area, caja, PCA (largos de ejes), bordes abiertos.
3. Clase:
   - astilla: < 4 caras o area < 20 mm2;
   - rueda: caja casi cuadrada en Y-Z (diametro >= 60 mm), delgada en X, en una esquina del auto;
   - tubo: eje principal >= 4 veces los otros dos y los dos menores parecidos (radio casi constante);
   - panel: lo demas.
4. Espejo en X: cada pieza con centro |x| > 3 mm busca la de -x con caras parecidas (+-15 %) y distancia
   simetrica media < 1.5 mm; la pieza de +x es la que se usa ("usar": true), la de -x se marca "espejo_de".
   Las piezas que cruzan X = 0 se usan tal cual.
"""
import json
import os
import sys
import hashlib
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, "..", ".."))
ENTRADA = os.path.join(RAIZ, "datos", "lowpoly", "a8bc0f28", "cuerpo.obj")
SAL = os.path.join(AQUI, "piezas")
os.makedirs(SAL, exist_ok=True)


def leer(ruta):
    V, F = [], []
    with open(ruta, encoding="utf-8", errors="ignore") as fh:
        for l in fh:
            if l.startswith("v "):
                V.append([float(x) for x in l.split()[1:4]])
            elif l.startswith("f "):
                F.append([int(x.split("/")[0]) - 1 for x in l.split()[1:]])
    return np.array(V), F


def soldar(V, tol=1e-3):
    q = np.round(V / tol).astype(np.int64)
    _, idx, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    return V[idx], inv.ravel()


def area_newell(P):
    n = np.zeros(3)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        n += np.cross(a, b)
    return 0.5 * np.linalg.norm(n)


def main():
    V0, F0 = leer(ENTRADA)
    Vs, inv = soldar(V0)
    F = [[int(inv[i]) for i in f] for f in F0]
    # componentes por arista compartida
    ed = defaultdict(list)
    for fi, f in enumerate(F):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            ed[(min(a, b), max(a, b))].append(fi)
    r, c = [], []
    for fs in ed.values():
        for i in range(1, len(fs)):
            r.append(fs[0]); c.append(fs[i])
    nF = len(F)
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(nF, nF))
    n_comp, lab = connected_components(A, directed=False)
    print("caras", nF, "vertices soldados", len(Vs), "de", len(V0), "componentes por arista", n_comp)

    largo_auto = V0[:, 1].max() - V0[:, 1].min()
    ymin, ymax = V0[:, 1].min(), V0[:, 1].max()
    piezas = []
    for k in range(n_comp):
        fids = np.where(lab == k)[0]
        vids = sorted({v for fi in fids for v in F[fi]})
        P = Vs[vids]
        cen = P.mean(0)
        lo, hi = P.min(0), P.max(0)
        dim = hi - lo
        area = sum(area_newell(Vs[F[fi]]) for fi in fids)
        ntri = sum(1 for fi in fids if len(F[fi]) == 3)
        ev = np.sort(np.linalg.eigvalsh(np.cov((P - cen).T))[::-1]) if len(P) >= 4 else np.array([1, 0, 0])
        ev = np.sqrt(np.maximum(ev[::-1], 0)) if len(P) >= 4 else np.array([1.0, 0, 0])
        ev = np.sort(ev)[::-1]
        borde = sum(1 for e, fs in ed.items() if len(fs) == 1 and lab[fs[0]] == k)
        nomani = sum(1 for e, fs in ed.items() if len(fs) > 2 and lab[fs[0]] == k)
        clase = "panel"
        if len(fids) < 4 or area < 20:
            clase = "astilla"
        else:
            yz = max(dim[1], dim[2])
            cerca_esquina = (min(abs(cen[1] - ymin), abs(cen[1] - ymax)) < 0.35 * largo_auto) and abs(cen[0]) > 60
            if (yz >= 60 and abs(dim[1] - dim[2]) < 0.15 * yz and dim[0] < 0.6 * yz and cerca_esquina
                    and cen[2] < 0.75 * yz):
                clase = "rueda"
            elif ev[0] >= 4 * max(ev[1], 1e-6) and ev[1] <= 1.6 * max(ev[2], 1e-6):
                clase = "tubo"
        piezas.append({"k": int(k), "caras": int(len(fids)), "triangulos": int(ntri),
                       "pct_tri": round(100.0 * ntri / len(fids), 1), "area_mm2": round(float(area), 1),
                       "centro": [round(float(x), 2) for x in cen], "min": [round(float(x), 2) for x in lo],
                       "max": [round(float(x), 2) for x in hi], "ejes_pca": [round(float(x), 2) for x in ev],
                       "aristas_borde": int(borde), "aristas_no_manifold": int(nomani), "clase": clase,
                       "_fids": fids, "_vids": vids})

    # lo que cae dentro de la caja de una rueda (rin, tapa, maza) tambien es rueda
    ruedas = [p for p in piezas if p["clase"] == "rueda"]
    for p in piezas:
        if p["clase"] in ("rueda",):
            continue
        for w in ruedas:
            lo, hi = np.array(w["min"]) - 3, np.array(w["max"]) + 3
            if np.all(np.array(p["min"]) >= lo) and np.all(np.array(p["max"]) <= hi):
                p["clase"] = "rueda"
                p["nota"] = "dentro de la caja de una rueda"
                break

    # ids estables: por clase y por posicion (frente a atras, izquierda a derecha)
    orden = sorted(range(len(piezas)), key=lambda i: (piezas[i]["centro"][1], piezas[i]["centro"][0], piezas[i]["centro"][2]))
    for n, i in enumerate(orden):
        piezas[i]["id"] = "s%03d" % n

    # espejo en X
    arboles = {}
    for p in piezas:
        P = Vs[p["_vids"]]
        arboles[p["id"]] = (P, cKDTree(P))
    por_id = {p["id"]: p for p in piezas}
    for p in piezas:
        p["usar"] = p["clase"] in ("panel", "tubo")
        p["espejo_de"] = None
        p["dist_espejo_mm"] = None
    cand = [p for p in piezas if p["clase"] in ("panel", "tubo", "rueda")]
    for p in cand:
        if p["centro"][0] <= 3:
            continue
        P, _ = arboles[p["id"]]
        Pm = P * np.array([-1, 1, 1])
        mejor = None
        for q in cand:
            if q is p or q["centro"][0] >= -3 or q["clase"] != p["clase"]:
                continue
            if abs(q["caras"] - p["caras"]) > 0.15 * max(p["caras"], q["caras"]) + 2:
                continue
            if np.linalg.norm(np.array(q["centro"]) - np.array(p["centro"]) * np.array([-1, 1, 1])) > 15:
                continue
            Q, tq = arboles[q["id"]]
            d1 = tq.query(Pm)[0]
            d2 = cKDTree(Pm).query(Q)[0]
            d = 0.5 * (d1.mean() + d2.mean())
            if mejor is None or d < mejor[0]:
                mejor = (d, q)
        if mejor and mejor[0] < 1.5:
            q = mejor[1]
            q["espejo_de"] = p["id"]
            q["usar"] = False
            q["dist_espejo_mm"] = round(float(mejor[0]), 3)
            p["dist_espejo_mm"] = round(float(mejor[0]), 3)

    # escribir OBJ por pieza (vertices soldados, orden de caras original dentro de la pieza)
    for p in piezas:
        mapa = {v: i for i, v in enumerate(p["_vids"])}
        ruta = os.path.join(SAL, p["id"] + ".obj")
        with open(ruta, "w", encoding="utf-8") as fh:
            fh.write("# Malla Smart a8bc0f28, pieza %s (componente %d), mm, Z arriba\n" % (p["id"], p["k"]))
            for v in p["_vids"]:
                x, y, z = Vs[v]
                fh.write("v %.6f %.6f %.6f\n" % (x, y, z))
            for fi in p["_fids"]:
                fh.write("f " + " ".join(str(mapa[v] + 1) for v in F[fi]) + "\n")
        with open(ruta, "rb") as fh:
            p["sha256"] = hashlib.sha256(fh.read()).hexdigest()
        del p["_fids"], p["_vids"]

    resumen = defaultdict(lambda: {"piezas": 0, "caras": 0, "usar": 0})
    for p in piezas:
        r = resumen[p["clase"]]
        r["piezas"] += 1; r["caras"] += p["caras"]; r["usar"] += int(p["usar"])
    with open(os.path.join(AQUI, "piezas.json"), "w", encoding="utf-8") as fh:
        json.dump({"entrada": "datos/lowpoly/a8bc0f28/cuerpo.obj", "caras": nF, "componentes": n_comp,
                   "resumen": resumen, "piezas": sorted(piezas, key=lambda p: p["id"])}, fh, indent=1, ensure_ascii=False)
    for k, v in resumen.items():
        print(k, dict(v))


if __name__ == "__main__":
    main()
