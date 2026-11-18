"""Referencia congelada v4.1 de un panel HD (solo lee datos/hd/pNN.obj).

Uso: python -I evaluacion/referencia_v4/extraer_referencia.py p68 [p37 ...]
Escribe evaluacion/referencia_v4/pNN.json.

Método (resumen, detalle en REVISION.md):
 1. Soldado de vértices (1e-3 mm), orientación coherente por componente (BFS).
 2. Filtro bilateral de normales (sigma espacial ~1.5 aristas, sigma angular 20°): quita el ruido
    de triángulos y deja normales por tramos.
 3. Crecimiento de regiones sobre normales filtradas (umbral contra la normal media de la región).
    Regiones diminutas se fusionan con la vecina de normal más parecida.
 4. Doblez = arista frontera entre dos regiones cuyo ángulo diedro medido a escala de mm
    (normales filtradas promediadas a DELTA mm a cada lado de la arista, dentro de cada región)
    es >= ANG_DOBLEZ. Los bordes abiertos nunca son doblez.
 5. Cadenas: unión de aristas de doblez, corte en cruces, suavizado ligero, puentes de huecos
    colineales <= 3 mm y tramos < 5 mm fuera. Fuera también las líneas pegadas al borde abierto
    (>= 70 % de su largo a < 2.5 mm del borde: canto o labio del panel, no doblez interior).
    La misma línea en una superficie opuesta (< 3 mm, normales opuestas) se deja una sola; dos
    líneas del mismo lado a < 2 mm (chaflán angosto) se funden en su línea media.
 6. Regiones planas: plano fijo por PCA ponderada en caras erosionadas (sin caras a menos de
    EROSION mm de un doblez o de un borde abierto); residuo p95/máx de la HD al plano. Solo es
    «plana» con residuo p95 <= 0.25 mm y >= 15 mm2; de 0.25 a 0.40 mm queda en casi_planas (no se usa).
 7. Reversos: pares de caras con normales opuestas (dot < -0.7) a 0.5-3 mm medidos sobre la normal
    y alineadas (separación lateral < 1.5 mm), no vecinas a 2 anillos. Los pares a < 0.5 mm se
    cuentan aparte como arrugas locales del triangulado.
"""
import json
import os
import sys
from collections import deque

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from malla_util import (Rejilla, aristas, geometria_caras, leer_obj, sha256,  # noqa: E402
                        soldar, triangular)

RAIZ = os.path.abspath(os.path.join(AQUI, "..", ".."))

PARAM = {
    "soldar_tol_mm": 1e-3,
    "bilateral_iter": 6,
    "bilateral_sigma_espacial_mm": 1.5,
    "bilateral_sigma_angular_deg": 20.0,
    "region_umbral_deg": 14.0,
    "region_area_min_mm2": 6.0,
    "delta_lado_mm": 1.5,
    "radio_lado_mm": 1.0,
    "ang_doblez_deg": 25.0,
    "largo_min_mm": 5.0,
    "puente_max_mm": 3.0,
    "puente_ang_deg": 25.0,
    "suavizado_iter": 2,
    "duplicado_dist_mm": 3.0,
    "fusion_dist_mm": 2.0,
    "duplicado_frac": 0.7,
    "duplicado_dot_opuesto": -0.5,
    "contorno_dist_mm": 2.5,
    "contorno_frac": 0.7,
    "relocalizar": False,  # probado el 17-nov: aleja 0.1-0.4 mm las líneas de los dobleces vivos (p41); queda apagado
    "reloc_radio_mm": 2.5,
    "reloc_sigma_mm": 1.0,
    "reloc_tau": 0.02,
    "reloc_suavizado_normal_mm": 0.7,
    "reloc_max_mm": 1.5,
    "erosion_mm": 1.5,
    "plana_residuo_p95_mm": 0.25,
    "casi_plana_residuo_p95_mm": 0.40,
    "dudoso_ang_min_deg": 18.0,
    "dudoso_largo_min_mm": 10.0,
    "plana_area_min_mm2": 15.0,
    "reverso_dist_mm": 3.0,
    "reverso_lateral_mm": 1.5,
    "reverso_dot": -0.7,
}


def orientar(T):
    """Orientación coherente por componente: devuelve T orientado y máscara de caras volteadas."""
    E, ef, cnt, fe = aristas(T)
    T = T.copy()
    nF = len(T)
    visto = np.zeros(nF, bool)
    volteada = np.zeros(nF, bool)
    comp = -np.ones(nF, int)
    vecinos = [[] for _ in range(nF)]
    for k, fs in enumerate(ef):
        if len(fs) == 2:
            vecinos[fs[0]].append((fs[1], k))
            vecinos[fs[1]].append((fs[0], k))

    def dir_arista(f, a, b):
        t = list(T[f])
        i = t.index(a)
        return t[(i + 1) % 3] == b

    nc = 0
    inconsistentes = 0
    for s in range(nF):
        if visto[s]:
            continue
        visto[s] = True
        comp[s] = nc
        q = deque([s])
        while q:
            f = q.popleft()
            for g, k in vecinos[f]:
                a, b = E[k]
                df = dir_arista(f, a, b)
                dg = dir_arista(g, a, b)
                if not visto[g]:
                    if df == dg:
                        T[g] = T[g][::-1]
                        volteada[g] = True
                    visto[g] = True
                    comp[g] = nc
                    q.append(g)
                elif df == dg:
                    inconsistentes += 1
        nc += 1
    return T, volteada, comp, nc, inconsistentes


def adyacencia_vertice(T, nV):
    """Pares (f, g) de caras que comparten al menos un vértice."""
    filas = np.repeat(np.arange(len(T)), 3)
    vs = T.reshape(-1)
    orden = np.argsort(vs, kind="stable")
    vs_o, f_o = vs[orden], filas[orden]
    cortes = np.flatnonzero(np.diff(vs_o)) + 1
    grupos = np.split(f_o, cortes)
    pares = set()
    for g in grupos:
        if len(g) > 1:
            for i in g:
                for j in g:
                    if i != j:
                        pares.add((i, j))
    P = np.array(sorted(pares), int)
    return P


def bilateral(n, area, cen, P, it, ss, sa_deg):
    nf = n.copy()
    d2 = np.sum((cen[P[:, 0]] - cen[P[:, 1]]) ** 2, axis=1)
    ws = area[P[:, 1]] * np.exp(-d2 / (2 * ss * ss))
    s1 = 1.0 - np.cos(np.radians(sa_deg))
    for _ in range(it):
        dot = np.sum(nf[P[:, 0]] * nf[P[:, 1]], axis=1)
        w = ws * np.exp(-((1.0 - dot) / s1) ** 2)
        acc = nf * area[:, None]
        np.add.at(acc, P[:, 0], nf[P[:, 1]] * w[:, None])
        nf = acc / np.maximum(np.linalg.norm(acc, axis=1), 1e-12)[:, None]
    return nf


def crecer_regiones(nf, area, vecinos_arista, umbral_deg, cen):
    nF = len(nf)
    reg = -np.ones(nF, int)
    cosu = np.cos(np.radians(umbral_deg))
    # semillas: caras cuya normal filtrada coincide con sus vecinas (más planas primero)
    var = np.zeros(nF)
    for f in range(nF):
        if vecinos_arista[f]:
            var[f] = 1.0 - np.mean([np.dot(nf[f], nf[g]) for g in vecinos_arista[f]])
    orden = np.argsort(var, kind="stable")
    r = 0
    for s in orden:
        if reg[s] >= 0:
            continue
        reg[s] = r
        suma = nf[s] * area[s]
        media = nf[s].copy()
        q = deque([s])
        while q:
            f = q.popleft()
            for g in vecinos_arista[f]:
                if reg[g] < 0 and np.dot(nf[g], media) >= cosu and np.dot(nf[g], nf[f]) >= np.cos(np.radians(umbral_deg * 0.6)):
                    reg[g] = r
                    suma += nf[g] * area[g]
                    media = suma / max(np.linalg.norm(suma), 1e-12)
                    q.append(g)
        r += 1
    return reg


def fusionar_pequenas(reg, nf, area, vecinos_arista, area_min):
    cambio = True
    while cambio:
        cambio = False
        ids = np.unique(reg)
        areas = {i: area[reg == i].sum() for i in ids}
        normales = {}
        for i in ids:
            s = (nf[reg == i] * area[reg == i][:, None]).sum(0)
            normales[i] = s / max(np.linalg.norm(s), 1e-12)
        for i in sorted(ids, key=lambda k: areas[k]):
            if areas[i] >= area_min:
                break
            caras = np.flatnonzero(reg == i)
            vec = set()
            for f in caras:
                for g in vecinos_arista[f]:
                    if reg[g] != i:
                        vec.add(reg[g])
            if not vec:
                continue
            mejor = max(vec, key=lambda j: np.dot(normales[j], normales[i]))
            reg[caras] = mejor
            cambio = True
            break
    # renumerar
    _, reg = np.unique(reg, return_inverse=True)
    return reg


def ajustar_plano(P, w):
    c = (P * w[:, None]).sum(0) / w.sum()
    X = (P - c) * np.sqrt(w)[:, None]
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    n = vt[-1]
    return n, c


def cadenas_de_aristas(Eidx):
    """Eidx: lista de pares de vértices. Devuelve cadenas (listas de vértices) cortadas en nodos de grado != 2."""
    ady = {}
    for a, b in Eidx:
        ady.setdefault(a, []).append(b)
        ady.setdefault(b, []).append(a)
    usadas = set()
    cadenas = []
    nodos = [v for v in ady if len(ady[v]) != 2]
    for v0 in nodos:
        for v1 in ady[v0]:
            k = frozenset((v0, v1))
            if k in usadas:
                continue
            usadas.add(k)
            cad = [v0, v1]
            prev, cur = v0, v1
            while len(ady[cur]) == 2:
                nxt = ady[cur][0] if ady[cur][0] != prev else ady[cur][1]
                k = frozenset((cur, nxt))
                if k in usadas:
                    break
                usadas.add(k)
                cad.append(nxt)
                prev, cur = cur, nxt
            cadenas.append(cad)
    # lazos cerrados
    for a, b in Eidx:
        k = frozenset((a, b))
        if k in usadas:
            continue
        usadas.add(k)
        cad = [a, b]
        prev, cur = a, b
        while True:
            nxt = ady[cur][0] if ady[cur][0] != prev else ady[cur][1]
            k = frozenset((cur, nxt))
            if k in usadas:
                break
            usadas.add(k)
            cad.append(nxt)
            prev, cur = cur, nxt
        cadenas.append(cad)
    return cadenas


def largo(P):
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) if len(P) > 1 else 0.0


def suavizar(P, it):
    P = P.copy()
    cerrada = len(P) > 3 and np.allclose(P[0], P[-1])
    for _ in range(it):
        Q = P.copy()
        Q[1:-1] = 0.5 * P[1:-1] + 0.25 * (P[:-2] + P[2:])
        P = Q
    if cerrada:
        P[-1] = P[0]
    return P


def remuestrear(P, paso):
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    if L[-1] < 1e-9:
        return P[:1]
    n = max(2, int(np.ceil(L[-1] / paso)) + 1)
    s = np.linspace(0, L[-1], n)
    return np.stack([np.interp(s, L, P[:, k]) for k in range(3)], axis=1)


def procesar(pid, param=PARAM):
    ruta = os.path.join(RAIZ, "datos", "hd", f"{pid}.obj")
    V0, F0 = leer_obj(ruta)
    T0, orig = triangular(F0)
    assert len(T0) == len(F0), "se esperaban solo triángulos"
    V, T, _ = soldar(V0, T0, param["soldar_tol_mm"])
    degeneradas = np.flatnonzero((T[:, 0] == T[:, 1]) | (T[:, 1] == T[:, 2]) | (T[:, 0] == T[:, 2]))
    n_raw, area0, _ = geometria_caras(V, T)
    degeneradas = np.union1d(degeneradas, np.flatnonzero(area0 < 1e-9))
    validas = np.setdiff1d(np.arange(len(T)), degeneradas)
    Tv = T[validas]

    To, volteada, comp, n_comp, inconsist = orientar(Tv)
    n, area, cen = geometria_caras(V, To)
    n_obj, _, _ = geometria_caras(V, Tv)  # normal según el orden del OBJ
    E, ef, cnt, fe = aristas(To)
    nF = len(To)
    vecinos = [[] for _ in range(nF)]
    for k, fs in enumerate(ef):
        if len(fs) == 2:
            vecinos[fs[0]].append(fs[1])
            vecinos[fs[1]].append(fs[0])
    borde = np.flatnonzero(cnt == 1)
    no_manifold = int((cnt > 2).sum())

    P = adyacencia_vertice(To, len(V))
    nf = bilateral(n, area, cen, P, param["bilateral_iter"], param["bilateral_sigma_espacial_mm"],
                   param["bilateral_sigma_angular_deg"])
    reg = crecer_regiones(nf, area, vecinos, param["region_umbral_deg"], cen)
    reg = fusionar_pequenas(reg, nf, area, vecinos, param["region_area_min_mm2"])
    n_reg = reg.max() + 1

    # --- diedro a escala de mm en aristas frontera entre regiones
    rej = Rejilla(cen, 1.0)
    frontera = [k for k in range(len(E)) if cnt[k] == 2 and reg[ef[k][0]] != reg[ef[k][1]]]
    d, rl = param["delta_lado_mm"], param["radio_lado_mm"]
    ang_e = {}
    for k in frontera:
        a, b = E[k]
        m = 0.5 * (V[a] + V[b])
        t = V[b] - V[a]
        t /= max(np.linalg.norm(t), 1e-12)
        lados = []
        for f in ef[k]:
            u = cen[f] - m
            u -= np.dot(u, t) * t
            u /= max(np.linalg.norm(u), 1e-12)
            p = m + d * u
            idx = rej.cerca(p, rl)
            idx = idx[reg[idx] == reg[f]]
            if len(idx) == 0:
                idx = np.array([f])
            s = (nf[idx] * area[idx][:, None]).sum(0)
            lados.append(s / max(np.linalg.norm(s), 1e-12))
        ang_e[k] = float(np.degrees(np.arccos(np.clip(np.dot(lados[0], lados[1]), -1, 1))))
    doblez_k = [k for k in frontera if ang_e[k] >= param["ang_doblez_deg"]]
    dud_k = [k for k in frontera if param["dudoso_ang_min_deg"] <= ang_e[k] < param["ang_doblez_deg"]]
    dudosos = []
    for cad in cadenas_de_aristas([tuple(map(int, E[k])) for k in dud_k]):
        Pd = suavizar(V[cad], param["suavizado_iter"])
        Ld = largo(Pd)
        if Ld >= param["dudoso_largo_min_mm"]:
            dudosos.append({"largo_mm": round(Ld, 2),
                            "puntos": [[round(float(c), 3) for c in q] for q in Pd]})
    clave = {frozenset(map(int, E[k])): k for k in doblez_k}

    cads = cadenas_de_aristas([tuple(map(int, E[k])) for k in doblez_k])
    lineas = []
    for cad in cads:
        Pts = V[cad]
        angs, ls = [], []
        for i in range(len(cad) - 1):
            k = clave[frozenset((cad[i], cad[i + 1]))]
            angs.append(ang_e[k])
            ls.append(np.linalg.norm(V[cad[i + 1]] - V[cad[i]]))
        Ps = suavizar(Pts, param["suavizado_iter"])
        L = largo(Ps)
        regs = sorted({int(reg[f]) for i in range(len(cad) - 1)
                       for f in ef[clave[frozenset((cad[i], cad[i + 1]))]]})
        lineas.append({"puntos": Ps, "largo": L, "ang": float(np.average(angs, weights=ls)),
                       "ang_max": float(np.max(angs)), "regiones": regs})

    # unir cadenas que se tocan en extremos con dirección continua (cortes en cruces débiles)
    lineas = unir_continuas(lineas)
    # puentes: tramos colineales separados por huecos cortos (el diedro baja un momento del umbral)
    lineas = puentear(lineas, param["puente_max_mm"], param["puente_ang_deg"])
    cortas = [ln for ln in lineas if ln["largo"] < param["largo_min_mm"]]
    lineas = [ln for ln in lineas if ln["largo"] >= param["largo_min_mm"]]

    # contornos: líneas pegadas al borde abierto (canto o labio del panel), no son dobleces interiores
    pts_borde = np.concatenate([remuestrear(V[[a, b]], 0.4) for a, b in E[borde]]) if len(borde) else np.zeros((0, 3))
    rej_b = Rejilla(pts_borde, param["contorno_dist_mm"]) if len(pts_borde) else None
    interiores, contornos = [], []
    for ln in lineas:
        Pm = remuestrear(ln["puntos"], 0.5)
        if rej_b is not None:
            cerca = np.array([len(rej_b.cerca(p, param["contorno_dist_mm"])) > 0 for p in Pm])
            ln["frac_cerca_borde"] = float(cerca.mean())
        else:
            ln["frac_cerca_borde"] = 0.0
        (contornos if ln["frac_cerca_borde"] >= param["contorno_frac"] else interiores).append(ln)
    lineas = interiores

    # normal media de cada región de la segmentación
    nreg = np.zeros((n_reg, 3))
    np.add.at(nreg, reg, nf * area[:, None])
    nreg /= np.maximum(np.linalg.norm(nreg, axis=1), 1e-12)[:, None]

    # duplicados y paralelas cercanas:
    #  - la misma línea en una superficie opuesta (reverso, normales opuestas, < 3 mm): se queda una sola;
    #  - dos líneas del mismo lado a < 2 mm (chaflán angosto de 1-2 mm): se funden en su línea media,
    #    porque a la tolerancia de C8 (0.5 mm) es un solo doblez.
    lineas.sort(key=lambda ln: -ln["largo"] * ln["ang"])
    finales, duplicadas, fundidas = [], [], []
    for ln in lineas:
        Pm = remuestrear(ln["puntos"], 0.5)
        destino = None
        for g in finales:
            Q = remuestrear(g["puntos"], 0.5)
            D = np.sqrt(((Pm[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
            dd = D.min(1)
            dmin = min(float(np.dot(nreg[i], nreg[j])) for i in ln["regiones"] for j in g["regiones"])
            if dmin <= param["duplicado_dot_opuesto"] and np.mean(dd <= param["duplicado_dist_mm"]) >= param["duplicado_frac"]:
                destino = ("dup", g)
                break
            if np.mean(dd <= param["fusion_dist_mm"]) >= param["duplicado_frac"]:
                destino = ("fusion", g, Q, D)
                break
        if destino is None:
            finales.append(ln)
        elif destino[0] == "dup":
            duplicadas.append(ln)
        else:
            _, g, Q, D = destino
            dq = D.min(0)
            jq = D.argmin(0)
            Qn = Q.copy()
            sel = dq <= param["fusion_dist_mm"]
            Qn[sel] = 0.5 * (Q[sel] + Pm[jq[sel]])
            g["puntos"] = suavizar(Qn, 2)
            g["largo"] = largo(g["puntos"])
            g["fundida_con"] = g.get("fundida_con", []) + [round(ln["largo"], 2)]
            g["ang_fundidas"] = g.get("ang_fundidas", [g["ang"]]) + [ln["ang"]]
            g["regiones"] = sorted(set(g["regiones"]) | set(ln["regiones"]))
            fundidas.append(ln)

    # --- relocalización: cada punto se lleva al centro del doblez (donde la normal local es la bisectriz)
    rel = Rejilla(cen, 1.0)
    ns = np.zeros_like(n)
    for f in (range(nF) if param["relocalizar"] else []):
        idx = rel.cerca(cen[f], param["reloc_suavizado_normal_mm"])
        if len(idx) == 0:
            idx = np.array([f])
        v = (n[idx] * area[idx][:, None]).sum(0)
        ns[f] = v / max(np.linalg.norm(v), 1e-12)
    for ln in (finales if param["relocalizar"] else []):
        P = remuestrear(ln["puntos"], 0.5)
        Q = P.copy()
        despl = []
        for i in range(len(P)):
            t = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
            t /= max(np.linalg.norm(t), 1e-12)
            idx = rel.cerca(P[i], param["reloc_radio_mm"])
            if len(idx) < 4:
                despl.append(0.0)
                continue
            nb = (ns[idx] * area[idx][:, None]).sum(0)
            nb /= max(np.linalg.norm(nb), 1e-12)
            d2 = np.sum((cen[idx] - P[i]) ** 2, axis=1)
            lw = -(1.0 - ns[idx] @ nb) / param["reloc_tau"] - d2 / (2 * param["reloc_sigma_mm"] ** 2)
            w = np.exp(lw - lw.max()) * area[idx]
            c = (cen[idx] * w[:, None]).sum(0) / w.sum()
            dv = c - P[i]
            dv -= np.dot(dv, t) * t
            dn = np.linalg.norm(dv)
            if dn > param["reloc_max_mm"]:
                dv *= param["reloc_max_mm"] / dn
            Q[i] = P[i] + dv
            despl.append(float(np.linalg.norm(dv)))
        ln["puntos"] = suavizar(Q, 3)
        ln["largo"] = largo(ln["puntos"])
        ln["reloc_media_mm"] = float(np.mean(despl)) if despl else 0.0
        ln["reloc_max_mm"] = float(np.max(despl)) if despl else 0.0

    # --- regiones planas con erosión
    pts_excl = [remuestrear(ln["puntos"], 0.4) for ln in finales]
    for k in borde:
        a, b = E[k]
        pts_excl.append(remuestrear(V[[a, b]], 0.4))
    pts_excl = np.concatenate(pts_excl) if pts_excl else np.zeros((0, 3))
    cerca_excl = np.zeros(nF, bool)
    if len(pts_excl):
        rej_e = Rejilla(pts_excl, param["erosion_mm"])
        for f in range(nF):
            if len(rej_e.cerca(cen[f], param["erosion_mm"])):
                cerca_excl[f] = True

    regiones = []
    casi_planas = []
    for r in range(n_reg):
        caras = np.flatnonzero((reg == r) & ~cerca_excl)
        if len(caras) < 3:
            continue
        a_r = area[caras].sum()
        if a_r < param["plana_area_min_mm2"]:
            continue
        vs = np.unique(To[caras].reshape(-1))
        Pv = np.concatenate([V[vs], cen[caras]])
        w = np.concatenate([np.full(len(vs), a_r / len(vs)), area[caras]])
        nn, c = ajustar_plano(Pv, w)
        nmedia = (n_obj[caras] * area[caras][:, None]).sum(0)
        if np.dot(nn, nmedia) < 0:
            nn = -nn
        dpl = -float(np.dot(nn, c))
        res = np.abs(V[vs] @ nn + dpl)
        p95 = float(np.percentile(res, 95))
        if p95 > param["plana_residuo_p95_mm"]:
            if p95 <= param["casi_plana_residuo_p95_mm"]:
                casi_planas.append({"region_segmentacion": int(r), "area_mm2": round(float(a_r), 2),
                                    "residuo_p95_mm": round(p95, 4),
                                    "plano": [float(nn[0]), float(nn[1]), float(nn[2]), dpl]})
            continue
        regiones.append({
            "id": len(regiones),
            "region_segmentacion": int(r),
            "plano": [float(nn[0]), float(nn[1]), float(nn[2]), dpl],
            "area_mm2": round(float(a_r), 2),
            "area_sin_erosion_mm2": round(float(area[reg == r].sum()), 2),
            "n_caras": int(len(caras)),
            "residuo_p95_mm": round(p95, 4),
            "residuo_max_mm": round(float(res.max()), 4),
            "caras": [int(validas[f]) for f in caras],
        })

    # --- reversos
    rev = reversos(V, To, n, area, cen, vecinos, comp, param, validas)

    # --- salida
    lin_json = []
    for i, ln in enumerate(sorted(finales, key=lambda x: -x["largo"])):
        lin_json.append({
            "id": i,
            "angulo_medio_deg": round(ln["ang"], 2),
            "angulo_max_deg": round(ln["ang_max"], 2),
            "largo_mm": round(ln["largo"], 2),
            "cerrada": bool(np.allclose(ln["puntos"][0], ln["puntos"][-1])),
            "regiones_segmentacion": ln["regiones"],
            "frac_a_menos_de_2_5mm_del_borde": round(ln["frac_cerca_borde"], 3),
            "puentes": int(ln.get("puentes", 0)),
            "relocalizacion_media_mm": round(ln.get("reloc_media_mm", 0.0), 3),
            "relocalizacion_max_mm": round(ln.get("reloc_max_mm", 0.0), 3),
            "fundida_con_largos_mm": ln.get("fundida_con", []),
            "angulos_fundidos_deg": [round(a, 2) for a in ln.get("ang_fundidas", [])],
            "puntos": [[round(float(x), 4) for x in p] for p in ln["puntos"]],
        })
    salida = {
        "contrato": "referencia_v4.1",
        "panel": pid,
        "fuente": f"datos/hd/{pid}.obj",
        "sha256_hd": sha256(ruta),
        "unidades": "mm, Z arriba",
        "parametros": param,
        "malla": {
            "vertices_obj": int(len(V0)), "caras_obj": int(len(F0)), "vertices_soldados": int(len(V)),
            "caras_degeneradas": [int(x) for x in degeneradas],
            "aristas_borde": int(len(borde)), "aristas_no_manifold": no_manifold,
            "componentes": int(n_comp), "caras_volteadas_para_orientar": int(volteada.sum()),
            "aristas_orientacion_inconsistente": int(inconsist),
            "area_mm2": round(float(area.sum()), 2),
            "regiones_segmentacion": int(n_reg),
        },
        "lineas_doblez": lin_json,
        "resumen_lineas": {
            "n": len(lin_json), "largo_total_mm": round(sum(x["largo_mm"] for x in lin_json), 2),
            "descartadas_cortas": len(cortas),
            "largo_descartado_corto_mm": round(sum(x["largo"] for x in cortas), 2),
            "descartadas_duplicadas": len(duplicadas),
            "largo_descartado_duplicado_mm": round(sum(x["largo"] for x in duplicadas), 2),
            "fundidas_en_otra": len(fundidas),
            "descartadas_contorno": len(contornos),
            "largo_descartado_contorno_mm": round(sum(x["largo"] for x in contornos), 2),
        },
        "descartadas": {
            "contorno": [{"largo_mm": round(x["largo"], 2), "angulo_medio_deg": round(x["ang"], 2),
                          "frac_cerca_borde": round(x["frac_cerca_borde"], 3),
                          "puntos": [[round(float(c), 3) for c in p] for p in x["puntos"]]} for x in contornos],
            "duplicadas": [{"largo_mm": round(x["largo"], 2), "angulo_medio_deg": round(x["ang"], 2),
                            "puntos": [[round(float(c), 3) for c in p] for p in x["puntos"]]} for x in duplicadas],
        },
        "regiones_planas": regiones,
        "resumen_regiones": {
            "n": len(regiones), "area_total_mm2": round(sum(r["area_mm2"] for r in regiones), 2),
            "fraccion_area_panel": round(sum(r["area_mm2"] for r in regiones) / float(area.sum()), 4),
        },
        "regiones_casi_planas_no_usadas": casi_planas,
        "dobleces_dudosos_no_usados": {
            "criterio": "fronteras entre regiones con diedro a escala de 18 a 25 grados y >= 10 mm: no entran a C8, quedan para revisión",
            "n": len(dudosos), "largo_total_mm": round(sum(d["largo_mm"] for d in dudosos), 2),
            "lineas": dudosos},
        "reversos": rev,
        "notas": [
            "Las líneas son interiores: ninguna arista de borde abierto cuenta como doblez.",
            "C8 se mide contra crease interiores del SubD; si lineas_doblez está vacío, C8 = N/D.",
            "C9: la SubD que cae (proyección más cercana a la HD) sobre 'caras' de cada región se mide contra 'plano' fijo.",
            "'caras' son índices base 0 en el orden original de las líneas f del OBJ, ya erosionadas.",
        ],
    }
    return salida, {"V": V, "T": To, "reg": reg, "borde_E": E[borde], "validas": validas}


def unir_continuas(lineas, tol=1e-6, ang_max=35.0):
    """Une cadenas que comparten extremo cuando en ese nodo solo se tocan dos y siguen derecho."""
    cambio = True
    while cambio:
        cambio = False
        extremos = {}
        for i, ln in enumerate(lineas):
            P = ln["puntos"]
            if np.allclose(P[0], P[-1]):
                continue
            for lado, p in ((0, P[0]), (1, P[-1])):
                extremos.setdefault(tuple(np.round(p, 4)), []).append((i, lado))
        for key, lst in extremos.items():
            if len(lst) != 2 or lst[0][0] == lst[1][0]:
                continue
            (i, li), (j, lj) = lst
            Pi = lineas[i]["puntos"] if li == 1 else lineas[i]["puntos"][::-1]
            Pj = lineas[j]["puntos"] if lj == 0 else lineas[j]["puntos"][::-1]
            di = Pi[-1] - Pi[max(0, len(Pi) - 4)]
            dj = Pj[min(len(Pj) - 1, 3)] - Pj[0]
            cosang = np.dot(di, dj) / max(np.linalg.norm(di) * np.linalg.norm(dj), 1e-12)
            if cosang < np.cos(np.radians(ang_max)):
                continue
            Li, Lj = lineas[i]["largo"], lineas[j]["largo"]
            nuevo = {
                "puntos": np.concatenate([Pi, Pj[1:]]),
                "largo": Li + Lj,
                "ang": (lineas[i]["ang"] * Li + lineas[j]["ang"] * Lj) / max(Li + Lj, 1e-9),
                "ang_max": max(lineas[i]["ang_max"], lineas[j]["ang_max"]),
                "regiones": sorted(set(lineas[i]["regiones"]) | set(lineas[j]["regiones"])),
            }
            lineas = [ln for k, ln in enumerate(lineas) if k not in (i, j)] + [nuevo]
            cambio = True
            break
    return lineas


def _dir_extremo(P, final):
    if final:
        d = P[-1] - P[max(0, len(P) - 5)]
    else:
        d = P[0] - P[min(len(P) - 1, 4)]
    return d / max(np.linalg.norm(d), 1e-12)


def puentear(lineas, gap, ang):
    cosa = np.cos(np.radians(ang))
    cambio = True
    while cambio:
        cambio = False
        mejor = None
        for i, a in enumerate(lineas):
            for j, b in enumerate(lineas):
                if i >= j:
                    continue
                for fa in (True, False):
                    for fb in (True, False):
                        pa = a["puntos"][-1] if fa else a["puntos"][0]
                        pb = b["puntos"][-1] if fb else b["puntos"][0]
                        g = pb - pa
                        dg = np.linalg.norm(g)
                        if dg > gap or dg < 1e-9:
                            continue
                        da = _dir_extremo(a["puntos"], fa)
                        db = _dir_extremo(b["puntos"], fb)
                        gu = g / dg
                        if np.dot(da, -db) >= cosa and np.dot(da, gu) >= cosa * 0.9 and np.dot(-db, gu) >= cosa * 0.9:
                            if mejor is None or dg < mejor[0]:
                                mejor = (dg, i, j, fa, fb)
        if mejor:
            _, i, j, fa, fb = mejor
            a, b = lineas[i], lineas[j]
            Pa = a["puntos"] if fa else a["puntos"][::-1]
            Pb = b["puntos"][::-1] if fb else b["puntos"]
            nuevo = {
                "puntos": np.concatenate([Pa, Pb]),
                "largo": 0.0,
                "ang": (a["ang"] * a["largo"] + b["ang"] * b["largo"]) / max(a["largo"] + b["largo"], 1e-9),
                "ang_max": max(a["ang_max"], b["ang_max"]),
                "regiones": sorted(set(a["regiones"]) | set(b["regiones"])),
                "puentes": a.get("puentes", 0) + b.get("puentes", 0) + 1,
            }
            nuevo["largo"] = largo(nuevo["puntos"])
            lineas = [ln for k, ln in enumerate(lineas) if k not in (i, j)] + [nuevo]
            cambio = True
    return lineas


def reversos(V, T, n, area, cen, vecinos, comp, param, validas=None):
    rej = Rejilla(cen, param["reverso_dist_mm"])
    nF = len(T)
    pares = []
    for f in range(nF):
        idx = rej.cerca(cen[f], param["reverso_dist_mm"])
        idx = idx[idx > f]
        if not len(idx):
            continue
        dot = n[idx] @ n[f]
        idx = idx[dot < param["reverso_dot"]]
        if not len(idx):
            continue
        dv = cen[idx] - cen[f]
        normal = dv @ n[f]
        lat = np.linalg.norm(dv - normal[:, None] * n[f], axis=1)
        idx = idx[(lat < param["reverso_lateral_mm"]) & (np.abs(normal) > 0.05)]
        for g in idx:
            # no vecinas geodésicas (dos anillos)
            v1 = set(vecinos[f])
            v2 = set(x for y in v1 for x in vecinos[y])
            if g in v1 or g in v2:
                continue
            pares.append((f, int(g), float(np.dot(cen[g] - cen[f], n[f]))))
    if not pares:
        return {"hay": False, "n_pares": 0, "texto": "No hay pares de caras con normales opuestas a < 3 mm."}
    sep_all = np.array([abs(p[2]) for p in pares])
    locales = [p for p in pares if abs(p[2]) < 0.5]
    caras_loc = sorted({p[0] for p in locales} | {p[1] for p in locales})
    pares = [p for p in pares if abs(p[2]) >= 0.5]
    resumen_locales = {"n_pares": len(locales), "n_caras": len(caras_loc),
                       "area_mm2": round(float(area[caras_loc].sum()), 2) if caras_loc else 0.0,
                       "nota": "pares a < 0.5 mm sobre la normal: arrugas o pliegues locales del triangulado, no una segunda superficie"}
    if not pares:
        return {"hay": False, "n_pares": 0, "pliegues_locales": resumen_locales,
                "texto": "No hay superficies opuestas separadas 0.5-3 mm; solo arrugas locales del triangulado."}
    caras = sorted({p[0] for p in pares} | {p[1] for p in pares})
    caras = np.array(caras)
    sep = np.array([abs(p[2]) for p in pares])
    # agrupar caras implicadas por conectividad
    grupo = -np.ones(nF, int)
    enc = set(caras.tolist())
    gid = 0
    grupos = []
    for s in caras:
        if grupo[s] >= 0:
            continue
        q = deque([s])
        grupo[s] = gid
        miembros = [s]
        while q:
            f = q.popleft()
            for g in vecinos[f]:
                if g in enc and grupo[g] < 0:
                    grupo[g] = gid
                    q.append(g)
                    miembros.append(g)
        miembros = np.array(miembros)
        grupos.append({
            "id": gid, "n_caras": int(len(miembros)),
            "area_mm2": round(float(area[miembros].sum()), 2),
            "centro": [round(float(x), 2) for x in cen[miembros].mean(0)],
            "bbox_min": [round(float(x), 2) for x in cen[miembros].min(0)],
            "bbox_max": [round(float(x), 2) for x in cen[miembros].max(0)],
            "componentes_malla": sorted({int(comp[m]) for m in miembros}),
            "caras": [int(validas[m]) if validas is not None else int(m) for m in sorted(miembros)],
        })
        gid += 1
    grupos.sort(key=lambda g: -g["area_mm2"])
    return {
        "hay": True,
        "n_pares": len(pares),
        "n_caras": int(len(caras)),
        "area_mm2": round(float(area[caras].sum()), 2),
        "fraccion_area_panel": round(float(area[caras].sum() / area.sum()), 4),
        "separacion_mm": {"min": round(float(sep.min()), 3), "mediana": round(float(np.median(sep)), 3),
                          "p95": round(float(np.percentile(sep, 95)), 3)},
        "grupos": grupos,
        "n_grupos": len(grupos),
        "pliegues_locales": resumen_locales,
        "criterio": "superficies opuestas = pares con separación 0.5-3 mm; dot(n_f, n_g) < -0.7, distancia sobre la normal en (0.05, 3] mm, separación lateral < 1.5 mm, no vecinas a 2 anillos; normales con orientación coherente por componente",
    }


def guardar_vis(pid, salida, extra):
    """Datos para el render: npz con V, T, regiones planas por cara, aristas de borde y líneas."""
    V, T, validas = extra["V"], extra["T"], extra["validas"]
    reg_plana = -np.ones(len(T), int)
    pos = {int(f): i for i, f in enumerate(validas)}
    for r in salida["regiones_planas"]:
        for f in r["caras"]:
            reg_plana[pos[f]] = r["id"]
    lineas = [np.array(ln["puntos"]) for ln in salida["lineas_doblez"]]
    reverso = np.zeros(len(T), bool)
    for g in salida["reversos"].get("grupos", []):
        for f in g["caras"]:
            if f in pos:
                reverso[pos[f]] = True
    np.savez(os.path.join(AQUI, "img", f"{pid}_vis.npz"), V=V, T=T, reg_plana=reg_plana, reverso=reverso,
             borde=extra["borde_E"], n_lineas=len(lineas), **{f"l{i}": l for i, l in enumerate(lineas)})


if __name__ == "__main__":
    ids = sys.argv[1:] or ["p68", "p37", "p41", "p30"]
    for pid in ids:
        sal, extra = procesar(pid)
        with open(os.path.join(AQUI, f"{pid}.json"), "w", encoding="utf-8") as fh:
            json.dump(sal, fh, indent=1, ensure_ascii=False)
        guardar_vis(pid, sal, extra)
        m = sal["malla"]
        rl, rr, rv = sal["resumen_lineas"], sal["resumen_regiones"], sal["reversos"]
        print(f"{pid}: regiones_seg={m['regiones_segmentacion']} lineas={rl['n']} largo={rl['largo_total_mm']} "
              f"(cortas {rl['descartadas_cortas']}, dup {rl['descartadas_duplicadas']}) | planas={rr['n']} "
              f"area={rr['area_total_mm2']} ({rr['fraccion_area_panel']*100:.0f}%) | reversos={rv.get('n_pares',0)} "
              f"caras={rv.get('n_caras',0)} area={rv.get('area_mm2',0)} | comps={m['componentes']} volteadas={m['caras_volteadas_para_orientar']} inconsist={m['aristas_orientacion_inconsistente']}")
        for ln in sal["lineas_doblez"]:
            print(f"   linea {ln['id']}: {ln['largo_mm']} mm, ang {ln['angulo_medio_deg']}")
