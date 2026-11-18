"""Referencia con tolerancia de bisel.

Uso: python evaluacion/referencia_smart_v5_w2/fundir_biseles.py [w_mm=2] [ids...]
Lee evaluacion/referencia_smart_v5/<id>.json (congelada, sin tocarla). Escribe evaluacion/referencia_smart_v5_w2/<id>.json.

Hecho que lo justifica: en s054, el 56 % del largo de las líneas de doblez tiene otra línea de doblez a menos de 2 mm.
Son las dos orillas de un bisel de Tripo. El criterio C8 exige las dos y el C3 prohíbe la tira entre ellas: se contradicen.
Con tolerancia w, un bisel más angosto que w es UNA línea de diseño: cada punto de una línea que tiene otra línea a
menos de w se mueve al punto medio entre él y el punto más cercano de la otra. Las dos orillas quedan sobre la misma
línea media (duplicada, lo que no cambia la cobertura de C8). Las regiones planas, esquinas y reversos no cambian.
"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree

AQUI = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(os.path.dirname(AQUI), "referencia_smart_v5")
W = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
IDS = sys.argv[2:] or [f[:-5] for f in sorted(os.listdir(ORIG)) if f.endswith(".json")]


def remuestrear(p, h=0.25):
    p = np.asarray(p, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    if len(p) < 2 or s[-1] < 1e-9:
        return p
    t = np.r_[np.arange(0, s[-1], h), s[-1]]
    return np.c_[[np.interp(t, s, p[:, k]) for k in range(3)]].T


for pid in IDS:
    d = json.load(open(os.path.join(ORIG, pid + ".json"), encoding="utf-8"))
    L = d.get("lineas_doblez", [])
    R = [remuestrear(l["puntos"]) for l in L]
    movidos = tot = 0
    if len(R) > 1:
        lab = np.concatenate([np.full(len(r), i) for i, r in enumerate(R)])
        A = np.concatenate(R)
        T = cKDTree(A)
        for i, (l, r) in enumerate(zip(L, R)):
            nuevo = []
            for q in r:
                tot += 1
                idx = [j for j in T.query_ball_point(q, W) if lab[j] != i]
                if idx:
                    j = min(idx, key=lambda j: np.linalg.norm(A[j] - q))
                    nuevo.append(((q + A[j]) / 2).tolist()); movidos += 1
                else:
                    nuevo.append(q.tolist())
            l["puntos"] = nuevo
            l["bisel_fundido_frac"] = None
    d["contrato"] = d.get("contrato", "referencia_v4.1")
    d["tolerancia_bisel_mm"] = W
    d["nota_tolerancia"] = ("Derivada de referencia_smart_v5 con fundir_biseles.py: las líneas a menos de %.1f mm de otra "
                            "se movieron a la línea media (%d de %d puntos remuestreados a 0.25 mm)." % (W, movidos, tot))
    json.dump(d, open(os.path.join(AQUI, pid + ".json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(pid, "puntos movidos %d de %d (%.0f%%)" % (movidos, tot, 100 * movidos / max(tot, 1)))
