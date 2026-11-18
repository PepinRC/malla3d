"""Trazado guiado (18-nov, 12:45): candidatas a líneas principales del frente +X de s054.
Uso: python salidas/guiado_v5/candidatas.py
Lee evaluacion/referencia_smart_v5_w2/s054.json (biseles < 2 mm ya fundidos en su línea media) y datos/smart_v5/piezas/s054.obj.
1. Zona: caras con centro en y <= YMAX y x >= -0.5 (frente, mitad +X).
2. Líneas de la zona (más de la mitad de sus puntos dentro): se quitan duplicados (las dos orillas fundidas de un bisel
   quedan casi encimadas: si el 80 % de una línea cae a < 0.6 mm de otra más larga, se descarta).
3. Escribe zona_frente.obj (las caras de la zona), candidatas.json (id, largo, ángulo medio, puntos) para elegir a ojo.
"""
import json, os
import numpy as np
from scipy.spatial import cKDTree

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, "..", ".."))
YMAX = -130.0

V, F = [], []
for l in open(os.path.join(RAIZ, "datos", "smart_v5", "piezas", "s054.obj"), encoding="utf-8"):
    if l.startswith("v "):
        V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith("f "):
        F.append([int(x) - 1 for x in l.split()[1:]])
V = np.array(V)
cen = np.array([V[f].mean(0) for f in F])
sel = [i for i in range(len(F)) if cen[i, 1] <= YMAX and cen[i, 0] >= -0.5]
usados = sorted({v for i in sel for v in F[i]})
mapa = {v: k for k, v in enumerate(usados)}
with open(os.path.join(AQUI, "zona_frente.obj"), "w", encoding="utf-8") as fh:
    fh.write("# s054, frente +X (y <= %.0f, x >= -0.5), %d caras de la Malla Smart\n" % (YMAX, len(sel)))
    for v in usados:
        fh.write("v %.6f %.6f %.6f\n" % tuple(V[v]))
    for i in sel:
        fh.write("f " + " ".join(str(mapa[v] + 1) for v in F[i]) + "\n")

d = json.load(open(os.path.join(RAIZ, "evaluacion", "referencia_smart_v5_w2", "s054.json"), encoding="utf-8"))
lin = []
for l in d["lineas_doblez"]:
    P = np.array(l["puntos"])
    if len(P) < 2:
        continue
    dentro = ((P[:, 1] <= YMAX + 5) & (P[:, 0] >= -1)).mean()
    if dentro > 0.5:
        largo = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        lin.append({"id_ref": l["id"], "largo_mm": round(largo, 1), "angulo_deg": l.get("angulo_medio_deg"), "P": P})
lin.sort(key=lambda x: -x["largo_mm"])
quedan = []
for l in lin:
    dup = False
    for q in quedan:
        dd = cKDTree(q["P"]).query(l["P"])[0]
        if (dd < 0.6).mean() >= 0.8:
            dup = True
            break
    if not dup:
        quedan.append(l)
for k, l in enumerate(quedan):
    l["n"] = k + 1
json.dump([{"n": l["n"], "id_ref": l["id_ref"], "largo_mm": l["largo_mm"], "angulo_deg": l["angulo_deg"],
            "puntos": l["P"].round(4).tolist()} for l in quedan],
          open(os.path.join(AQUI, "candidatas.json"), "w", encoding="utf-8"), indent=1)
print("zona", len(sel), "caras; lineas", len(lin), "-> sin duplicados", len(quedan))
for l in quedan:
    print(l["n"], l["id_ref"], l["largo_mm"], l["angulo_deg"])
