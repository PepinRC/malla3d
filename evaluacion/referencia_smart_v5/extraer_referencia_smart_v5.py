"""Referencia congelada para la Malla Smart de Tripo (variante a8bc0f28), contrato referencia_v4.1.

Uso: python evaluacion/referencia_smart_v5/extraer_referencia_smart_v5.py s054 s071 ...  (vuelta 5, 18-nov)
Lee SOLO datos/smart_v5/piezas/<id>.obj (piezas de la Malla Smart a8bc0f28 sin el HD). Escribe <id>.json e img/<id>_vis.npz.

Metodo (la malla ya viene por facetas, asi que no hay filtro bilateral ni segmentacion por ruido):
 1. Orientacion coherente por componente (BFS sobre aristas con 2 caras). Aristas con 3 caras no cuentan.
 2. Diedro con signo entre las dos caras de cada arista interior (normales de Newell de los poligonos).
 3. Doblez = arista interior con |diedro| >= 25. Los bordes abiertos nunca cuentan.
 4. Faceta de curva: si en AMBAS caras de la arista existe otra arista casi paralela (|cos| >= 0.8) cuyo diedro
    es del mismo sentido y esta entre 5 y 25 grados, la arista es un paso entre facetas de una curva suave y se
    excluye (`excluidas_curva`). Si una cara no tiene arista paralela (p. ej. un triangulo), no se puede decir que
    sea curva y la arista se queda como doblez.
 5. Cadenas: aristas de doblez unidas en polilineas, sin suavizado (las aristas ya son exactas); se unen las que
    siguen derechas (<= 35 grados) en un nodo de dos extremos; tramos < 5 mm fuera.
 6. Regiones planas: crecimiento desde la cara de mayor area; entra una vecina si su normal queda a < 3 grados del
    plano de la region y sus vertices a <= 0.2 mm; plano por PCA, recalculado cuando la region se duplica. Un cuadro
    alabeado (residuo propio > 0.2 mm) no es region. Cuentan las regiones con >= 1 % del area del panel (sin erosion).
    Las caras con el centro a < 1.5 mm de un doblez o de un borde abierto no van en `caras` (erosion de v4).
 7. Reversos: la misma funcion de referencia_v4 (pares de caras con normales opuestas a 0.5-3 mm).
"""
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from malla_util import Rejilla, aristas, geometria_caras, leer_obj, sha256  # noqa: E402
import extraer_referencia as v4  # noqa: E402  (copia de referencia_v4: reversos, cadenas, unir_continuas...)

RAIZ = os.path.abspath(os.path.join(AQUI, "..", ".."))
VARIANTE = "a8bc0f28"

PARAM = {
    "soldar_tol_mm": 1e-3,
    "ang_doblez_deg": 25.0,
    "largo_min_mm": 5.0,
    "curva_vecina_min_deg": 5.0,
    "curva_vecina_max_deg": 25.0,
    "curva_paralela_cos": 0.8,
    "unir_ang_max_deg": 35.0,
    "region_normal_deg": 3.0,
    "region_residuo_mm": 0.2,
    "plana_residuo_p95_mm": 0.2,
    "plana_area_min_frac": 0.01,
    "plana_area_min_mm2": 15.0,
    "erosion_mm": 1.5,
    "contorno_dist_mm": 2.5,
    "dudoso_ang_min_deg": 18.0,
    "dudoso_largo_min_mm": 10.0,
    "casi_plana_residuo_p95_mm": 0.4,
    "reverso_dist_mm": 3.0,
    "reverso_lateral_mm": 1.5,
    "reverso_dot": -0.7,
}


def newell(V, f):
    P = V[f]
    n = np.zeros(3)
    for i in range(len(f)):
        a, b = P[i], P[(i + 1) % len(f)]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    return n


def orientar_poligonos(F):
    """BFS por componente sobre aristas con exactamente 2 caras. Devuelve F orientada, comp, n_comp, inconsistentes, volteadas."""
    ed = {}
    for i, f in enumerate(F):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            ed.setdefault((min(a, b), max(a, b)), []).append(i)
    vec = [[] for _ in F]
    for (a, b), fs in ed.items():
        if len(fs) == 2:
            vec[fs[0]].append((fs[1], a, b))
            vec[fs[1]].append((fs[0], a, b))
    G = [list(f) for f in F]
    comp = -np.ones(len(F), int)
    volt = 0

    def dirc(f, a, b):
        i = G[f].index(a)
        return G[f][(i + 1) % len(G[f])] == b

    nc = 0
    for s in range(len(F)):
        if comp[s] >= 0:
            continue
        comp[s] = nc
        pila = [s]
        while pila:
            f = pila.pop()
            for g, a, b in vec[f]:
                if comp[g] < 0:
                    if dirc(f, a, b) == dirc(g, a, b):
                        G[g] = G[g][::-1]
                        volt += 1
                    comp[g] = nc
                    pila.append(g)
        nc += 1
    inc = 0
    for (a, b), fs in ed.items():
        if len(fs) == 2 and dirc(fs[0], a, b) == dirc(fs[1], a, b):
            inc += 1
    return G, comp, nc, inc, volt


def diedro(nA, nB, t):
    t = t / max(np.linalg.norm(t), 1e-12)
    return float(np.degrees(np.arctan2(np.dot(np.cross(nA, nB), t), np.dot(nA, nB))))


def fit_plano(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return vt[-1], c


def procesar(pid, param=PARAM):
    ruta = os.path.join(RAIZ, "datos", "smart_v5", "piezas", f"{pid}.obj")
    V, F0 = leer_obj(ruta)
    nF = len(F0)
    n_obj = np.array([newell(V, f) for f in F0])
    n_obj /= np.maximum(np.linalg.norm(n_obj, axis=1), 1e-15)[:, None]
    F, comp, n_comp, inconsist, volteadas = orientar_poligonos(F0)
    nn = np.array([newell(V, f) for f in F])
    area = 0.5 * np.linalg.norm(nn, axis=1)
    n = nn / np.maximum(2 * area, 1e-15)[:, None]
    cen = np.array([V[f].mean(0) for f in F])
    A_tot = float(area.sum())

    ed = {}
    for i, f in enumerate(F):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            ed.setdefault((min(a, b), max(a, b)), []).append((i, a, b))
    borde = [k for k, v in ed.items() if len(v) == 1]
    no_manifold = [k for k, v in ed.items() if len(v) > 2]
    vecinos = [[] for _ in range(nF)]
    for k, v in ed.items():
        if len(v) == 2:
            vecinos[v[0][0]].append(v[1][0])
            vecinos[v[1][0]].append(v[0][0])

    # diedro con signo por arista interior
    sd = {}
    for k, v in ed.items():
        if len(v) != 2:
            continue
        (fa, a, b), (fb, _, _) = v
        sd[k] = diedro(n[fa], n[fb], V[b] - V[a])

    def vecina_paralela(f, k):
        """Diedro (con signo) de la arista de la cara f mas paralela a la arista k, o None."""
        a0, b0 = k
        t0 = V[b0] - V[a0]
        t0 /= max(np.linalg.norm(t0), 1e-12)
        mejor = None
        for j in range(len(F[f])):
            a, b = F[f][j], F[f][(j + 1) % len(F[f])]
            kk = (min(a, b), max(a, b))
            if kk == k:
                continue
            t = V[b] - V[a]
            t /= max(np.linalg.norm(t), 1e-12)
            c = abs(float(np.dot(t, t0)))
            if c >= param["curva_paralela_cos"] and kk in sd and (mejor is None or c > mejor[0]):
                mejor = (c, sd[kk], kk)
        return None if mejor is None else (mejor[1], mejor[2])

    doblez, curva, dud = [], [], []
    for k, s in sd.items():
        if abs(s) >= param["ang_doblez_deg"]:
            (fa, _, _), (fb, _, _) = ed[k]
            lado = [vecina_paralela(f, k) for f in (fa, fb)]
            es_curva = all(r is not None and np.sign(r[0]) == np.sign(s)
                           and param["curva_vecina_min_deg"] <= abs(r[0]) < param["curva_vecina_max_deg"]
                           for r in lado)
            if es_curva:
                curva.append((k, s, [round(r[0], 1) for r in lado]))
            else:
                doblez.append(k)
        elif param["dudoso_ang_min_deg"] <= abs(s) < param["ang_doblez_deg"]:
            dud.append(k)

    # --- regiones coplanares (todas las caras) ---
    asig = -np.ones(nF, int)
    grupos = []

    def verts(fs):
        return np.unique(np.concatenate([F[f] for f in fs]))

    def res_vertices(nrm, c, vs):
        return np.abs((V[vs] - c) @ nrm)

    cos3 = np.cos(np.radians(param["region_normal_deg"]))
    pendientes = list(np.argsort(-area, kind="stable"))
    while pendientes:
        s = int(pendientes.pop(0))
        if asig[s] >= 0:
            continue
        gid = len(grupos)
        nrm, c = fit_plano(V[F[s]])
        if np.dot(nrm, n[s]) < 0:
            nrm = -nrm
        if res_vertices(nrm, c, np.array(F[s])).max() > param["region_residuo_mm"]:
            asig[s] = gid
            grupos.append({"caras": [s], "plano": None, "alabeada": True})
            continue
        miembros = [s]
        asig[s] = gid
        ult = 1
        cola = [s]
        while cola:
            f = cola.pop(0)
            for g in vecinos[f]:
                if asig[g] >= 0:
                    continue
                if np.dot(n[g], nrm) < cos3:
                    continue
                if res_vertices(nrm, c, np.array(F[g])).max() > param["region_residuo_mm"]:
                    continue
                asig[g] = gid
                miembros.append(int(g))
                cola.append(int(g))
                if len(miembros) >= 2 * ult:
                    nrm, c = fit_plano(V[verts(miembros)])
                    if np.dot(nrm, n[s]) < 0:
                        nrm = -nrm
                    ult = len(miembros)
        for _ in range(10):
            nrm, c = fit_plano(V[verts(miembros)])
            if np.dot(nrm, (n[miembros] * area[miembros][:, None]).sum(0)) < 0:
                nrm = -nrm
            malas = [m for m in miembros if m != s and
                     (res_vertices(nrm, c, np.array(F[m])).max() > param["region_residuo_mm"] or np.dot(n[m], nrm) < cos3)]
            if not malas:
                break
            for m in malas:
                miembros.remove(m)
                asig[m] = -1
                pendientes.insert(0, m)
        grupos.append({"caras": miembros, "plano": (nrm, c), "alabeada": False})
    n_grupos = len(grupos)

    # --- cadenas de doblez ---
    def lineas_de(keys):
        out = []
        for cad in v4.cadenas_de_aristas([tuple(map(int, k)) for k in keys]):
            Pts = V[cad]
            angs, ls, regs = [], [], set()
            for i in range(len(cad) - 1):
                k = (min(cad[i], cad[i + 1]), max(cad[i], cad[i + 1]))
                angs.append(abs(sd[k]))
                ls.append(float(np.linalg.norm(V[cad[i + 1]] - V[cad[i]])))
                for f, _, _ in ed[k]:
                    regs.add(int(asig[f]))
            out.append({"puntos": Pts, "largo": v4.largo(Pts),
                        "ang": float(np.average(angs, weights=np.maximum(ls, 1e-9))),
                        "ang_max": float(max(angs)), "regiones": sorted(regs)})
        return out

    lineas = lineas_de(doblez)
    lineas = v4.unir_continuas(lineas, ang_max=param["unir_ang_max_deg"])
    cortas = [ln for ln in lineas if ln["largo"] < param["largo_min_mm"]]
    lineas = [ln for ln in lineas if ln["largo"] >= param["largo_min_mm"]]
    lineas.sort(key=lambda ln: -ln["largo"])

    pts_borde = (np.concatenate([v4.remuestrear(V[list(k)], 0.4) for k in borde]) if borde else np.zeros((0, 3)))
    rej_b = Rejilla(pts_borde, param["contorno_dist_mm"]) if len(pts_borde) else None
    for ln in lineas:
        Pm = v4.remuestrear(ln["puntos"], 0.5)
        ln["frac_cerca_borde"] = (float(np.mean([len(rej_b.cerca(p, param["contorno_dist_mm"])) > 0 for p in Pm]))
                                  if rej_b is not None else 0.0)

    dudosos = []
    for ln in lineas_de(dud):
        if ln["largo"] >= param["dudoso_largo_min_mm"]:
            dudosos.append({"largo_mm": round(ln["largo"], 2),
                            "puntos": [[round(float(c), 3) for c in p] for p in ln["puntos"]]})
    cur_json = []
    sdc = {k: (s, vec) for k, s, vec in curva}
    for cad in v4.cadenas_de_aristas([tuple(map(int, k)) for k, _, _ in curva]):
        ks = [(min(cad[i], cad[i + 1]), max(cad[i], cad[i + 1])) for i in range(len(cad) - 1)]
        Pts = V[cad]
        cur_json.append({
            "largo_mm": round(v4.largo(Pts), 2),
            "diedros_deg": [round(sdc[k][0], 1) for k in ks],
            "diedros_vecinas_deg": [sdc[k][1] for k in ks],
            "puntos": [[round(float(c), 4) for c in p] for p in Pts],
        })
    cur_json.sort(key=lambda d: -d["largo_mm"])

    # --- regiones planas con erosion ---
    pts_excl = [v4.remuestrear(ln["puntos"], 0.4) for ln in lineas]
    pts_excl += [v4.remuestrear(V[list(k)], 0.4) for k in borde]
    pts_excl = np.concatenate(pts_excl) if pts_excl else np.zeros((0, 3))
    cerca_excl = np.zeros(nF, bool)
    if len(pts_excl):
        rej_e = Rejilla(pts_excl, param["erosion_mm"])
        for f in range(nF):
            if len(rej_e.cerca(cen[f], param["erosion_mm"])):
                cerca_excl[f] = True

    def aristas_cara(f):
        return [(min(F[f][j], F[f][(j + 1) % len(F[f])]), max(F[f][j], F[f][(j + 1) % len(F[f])]))
                for j in range(len(F[f]))]

    regiones, casi, alab = [], [], []
    for g, gr in enumerate(grupos):
        A_g = float(area[gr["caras"]].sum())
        if A_g < param["plana_area_min_frac"] * A_tot:
            continue
        if gr["alabeada"]:
            f = gr["caras"][0]
            nrm, c = fit_plano(V[F[f]])
            r = float(np.percentile(res_vertices(nrm, c, np.array(F[f])), 95))
            (casi if r <= param["casi_plana_residuo_p95_mm"] else alab).append(
                {"cara": int(f), "area_mm2": round(A_g, 2), "residuo_p95_mm": round(r, 3)})
            continue
        caras = [f for f in gr["caras"] if not cerca_excl[f]]
        if not caras or float(area[caras].sum()) < param["plana_area_min_mm2"]:
            continue
        nrm, c = gr["plano"]
        nmedia = (n_obj[caras] * area[caras][:, None]).sum(0)
        if np.dot(nrm, nmedia) < 0:
            nrm = -nrm
        res = res_vertices(nrm, c, verts(caras))
        p95 = float(np.percentile(res, 95))
        if p95 > param["plana_residuo_p95_mm"]:
            casi.append({"region": g, "area_mm2": round(A_g, 2), "residuo_p95_mm": round(p95, 4)})
            continue
        faceta = all(any(k in sdc for k in aristas_cara(f)) for f in gr["caras"])
        regiones.append({
            "id": len(regiones),
            "region_segmentacion": int(g),
            "plano": [float(nrm[0]), float(nrm[1]), float(nrm[2]), -float(np.dot(nrm, c))],
            "area_mm2": round(float(area[caras].sum()), 2),
            "area_sin_erosion_mm2": round(A_g, 2),
            "n_caras": len(caras),
            "residuo_p95_mm": round(p95, 4),
            "residuo_max_mm": round(float(res.max()), 4),
            "faceta_de_curva": bool(faceta),
            "caras": sorted(int(f) for f in caras),
        })

    # --- reversos: misma funcion que v4, sobre los triangulos de los poligonos orientados ---
    T, orig = [], []
    for i, f in enumerate(F):
        for j in range(1, len(f) - 1):
            T.append((f[0], f[j], f[j + 1]))
            orig.append(i)
    T = np.array(T, int)
    orig = np.array(orig, int)
    nt, at, ct = geometria_caras(V, T)
    _, eft, _, _ = aristas(T)
    vect = [[] for _ in range(len(T))]
    for fs in eft:
        if len(fs) == 2:
            vect[fs[0]].append(fs[1])
            vect[fs[1]].append(fs[0])
    rev = v4.reversos(V, T, nt, at, ct, vect, comp[orig], param, orig)
    for g in rev.get("grupos", []):
        g["caras"] = sorted(set(g["caras"]))
        g["n_caras"] = len(g["caras"])

    lin_json = []
    for i, ln in enumerate(lineas):
        lin_json.append({
            "id": i,
            "angulo_medio_deg": round(ln["ang"], 2),
            "angulo_max_deg": round(ln["ang_max"], 2),
            "largo_mm": round(ln["largo"], 2),
            "cerrada": bool(np.allclose(ln["puntos"][0], ln["puntos"][-1])),
            "regiones_segmentacion": ln["regiones"],
            "frac_a_menos_de_2_5mm_del_borde": round(ln["frac_cerca_borde"], 3),
            "puentes": 0,
            "relocalizacion_media_mm": 0.0,
            "relocalizacion_max_mm": 0.0,
            "fundida_con_largos_mm": [],
            "angulos_fundidos_deg": [],
            "puntos": [[round(float(x), 4) for x in p] for p in ln["puntos"]],
        })
    sal = {
        "contrato": "referencia_v4.1",
        "panel": pid,
        "entrada": f"Malla Smart de Tripo, variante {VARIANTE}",
        "fuente": f"datos/smart_v5/piezas/{pid}.obj",
        "sha256_hd": sha256(ruta),
        "unidades": "mm, Z arriba",
        "parametros": param,
        "malla": {
            "vertices_obj": int(len(V)), "caras_obj": int(nF),
            "caras_cuadros": int(sum(len(f) == 4 for f in F0)), "caras_triangulos": int(sum(len(f) == 3 for f in F0)),
            "vertices_soldados": int(len(V)), "caras_degeneradas": [],
            "aristas_borde": len(borde), "aristas_no_manifold": len(no_manifold),
            "componentes": int(n_comp), "caras_volteadas_para_orientar": int(volteadas),
            "aristas_orientacion_inconsistente": int(inconsist),
            "area_mm2": round(A_tot, 2),
            "regiones_segmentacion": int(n_grupos),
        },
        "lineas_doblez": lin_json,
        "resumen_lineas": {
            "n": len(lin_json), "largo_total_mm": round(sum(x["largo_mm"] for x in lin_json), 2),
            "aristas_candidatas_25deg": len(doblez) + len(curva),
            "aristas_en_lineas": len(doblez),
            "descartadas_cortas": len(cortas),
            "largo_descartado_corto_mm": round(sum(x["largo"] for x in cortas), 2),
            "lineas_pegadas_al_borde_no_descartadas": int(sum(1 for x in lin_json if x["frac_a_menos_de_2_5mm_del_borde"] >= 0.7)),
        },
        "excluidas_curva": {
            "criterio": "arista con diedro >= 25 grados entre facetas de una curva: en las dos caras hay otra arista casi paralela (|cos| >= 0.8) con diedro del mismo sentido entre 5 y 25 grados",
            "n_aristas": len(curva),
            "largo_total_mm": round(sum(d["largo_mm"] for d in cur_json), 2),
            "tramos": cur_json,
        },
        "regiones_planas": regiones,
        "resumen_regiones": {
            "n": len(regiones), "area_total_mm2": round(sum(r["area_mm2"] for r in regiones), 2),
            "fraccion_area_panel": round(sum(r["area_mm2"] for r in regiones) / A_tot, 4),
            "de_las_cuales_facetas_de_curva": int(sum(r["faceta_de_curva"] for r in regiones)),
        },
        "regiones_casi_planas_no_usadas": casi,
        "cuadros_alabeados_no_usados": alab,
        "dobleces_dudosos_no_usados": {
            "criterio": "aristas con diedro de 18 a 25 grados encadenadas, >= 10 mm: no entran a C8",
            "n": len(dudosos), "largo_total_mm": round(sum(d["largo_mm"] for d in dudosos), 2), "lineas": dudosos},
        "reversos": rev,
        "notas": [
            "Las lineas son interiores: ninguna arista de borde abierto cuenta como doblez.",
            "No hay suavizado, fusion de paralelas ni descarte por contorno: la entrada ya es de aristas exactas.",
            "C8 se mide contra crease interiores del SubD; si lineas_doblez esta vacio, C8 = N/D.",
            "C9: la SubD que cae sobre 'caras' de cada region se mide contra 'plano' fijo.",
            "'caras' son indices base 0 en el orden original de las lineas f del OBJ (cuadros y triangulos), ya erosionadas.",
        ],
    }
    extra = {"V": V, "F": F0, "borde": np.array(borde, int).reshape(-1, 2),
             "lineas": [np.array(l["puntos"]) for l in lin_json], "curva": [np.array(c["puntos"]) for c in cur_json],
             "int_edges": np.array([k for k, v in ed.items() if len(v) == 2], int).reshape(-1, 2),
             "rev_caras": sorted({c for g in rev.get("grupos", []) for c in g["caras"]})}
    return sal, extra


def guardar_vis(pid, sal, ex):
    V, F0 = ex["V"], ex["F"]
    nf = len(F0)
    reg_plana = -np.ones(nf, int)
    for r in sal["regiones_planas"]:
        for f in r["caras"]:
            reg_plana[f] = r["id"]
    rev = np.zeros(nf, bool)
    rev[ex["rev_caras"]] = True
    P = -np.ones((nf, 4), int)
    for i, f in enumerate(F0):
        P[i, :len(f)] = f
    np.savez(os.path.join(AQUI, "img", f"{pid}_vis.npz"), V=V, P=P, reg_plana=reg_plana, reverso=rev,
             borde=ex["borde"], int_edges=ex["int_edges"], n_lineas=len(ex["lineas"]), n_curva=len(ex["curva"]),
             **{f"l{i}": l for i, l in enumerate(ex["lineas"])}, **{f"c{i}": l for i, l in enumerate(ex["curva"])})


if __name__ == "__main__":
    ids = sys.argv[1:] or ["p68", "p37", "p41", "p30"]
    for pid in ids:
        sal, ex = procesar(pid)
        with open(os.path.join(AQUI, f"{pid}.json"), "w", encoding="utf-8") as fh:
            json.dump(sal, fh, indent=1, ensure_ascii=False)
        guardar_vis(pid, sal, ex)
        m, rl, rr, rv, ec = sal["malla"], sal["resumen_lineas"], sal["resumen_regiones"], sal["reversos"], sal["excluidas_curva"]
        print(f"{pid}: caras={m['caras_obj']} borde={m['aristas_borde']} nomanif={m['aristas_no_manifold']} inconsist={m['aristas_orientacion_inconsistente']} "
              f"| cand25={rl['aristas_candidatas_25deg']} lineas={rl['n']} largo={rl['largo_total_mm']} cortas={rl['descartadas_cortas']} "
              f"curva={ec['n_aristas']}ar/{ec['largo_total_mm']}mm | planas={rr['n']} area={rr['area_total_mm2']} ({rr['fraccion_area_panel']*100:.0f}%) facetas_curva={rr['de_las_cuales_facetas_de_curva']} "
              f"| reversos={rv.get('n_pares',0)} caras={rv.get('n_caras',0)}")
        for ln in sal["lineas_doblez"]:
            print(f"   linea {ln['id']}: {ln['largo_mm']} mm ang {ln['angulo_medio_deg']} borde {ln['frac_a_menos_de_2_5mm_del_borde']}")
