#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Banco de evaluación y generación de imágenes en Blender 5.2 para Malla3D (Vuelta 2).

Uso CLI:
  blender.exe -b --factory-startup -P evaluacion/evaluar_blender.py -- \
      --jaulas <carpeta> [<carpeta2> ...] \
      --ref datos/hd \
      --salida <carpeta_resultados> \
      [--lineas salidas/lineas] \
      [--crease-auto 30] \
      [--solo p68,p37,p41,p30] \
      [--referencia-v4 evaluacion/referencia_v4]

  --solo: evalúa solo esas piezas (las demás se ignoran, también en hojas e imágenes).
  --referencia-v4: contrato referencia_v4.1 (17-nov-2026, ver evaluacion/CAMBIOS.md). Las líneas y
    regiones salen SOLO de <carpeta>/pNN.json (congeladas de la HD); C8 contra crease interiores
    de la SubD (los bordes no cuentan, sin líneas = N/D); C9 contra el plano fijo de cada región
    (peor p95 por región, sin recentrar); C3 con la proporción bruta; se reporta la media inversa
    HD->SubD. Escribe resultados_v4.json y resumen_v4.md (no resultados.json ni resumen.md).

Criterios evaluados por panel:
  1. Cuadros: 0 % de triángulos en la jaula.
  2. Ligera: de 10 a 150 caras por panel.
  3. Sin caras estiradas: proporción (arista mayor / arista menor) > 3.5 en <= 10 % de caras.
  4. Sin polos raros: 0 vértices interiores con 6 o más aristas.
  5. Fiel: SubD (Catmull-Clark nivel 3) -> HD media <= 0.3 mm, p95 <= 1.5 mm, y HD -> SubD p95 <= 1.5 mm.
  6-7. Paneles planos y aristas vivas donde el diseño dobla.
  8. Aristas del diseño respetadas: las líneas de doblez de la referencia HD (donde la superficie
     dobla más de 25-30°, extraídas de salidas/lineas/pNN.json si existe o directamente de datos/hd)
     caen sobre aristas vivas de la superficie subdividida: p95 <= 0.5 mm y >= 90 % del largo cubierto.
  9. Caras planas sin ondas (planitud): en cada región plana de la referencia, la superficie
     subdividida se desvía de su plano <= 0.2 mm en el p95.

Imágenes generadas:
  - Por pieza y frente completo desde 3 cámaras fijas (34, costado, frente) en alambre y sombreado.
  - De cerca por panel con luz rasante: <pieza>_alambre_cerca.png, <pieza>_sombreado_cerca.png y
    la composición lado a lado <pieza>_cerca.png (alambre + sombreado con luz rasante para ver ondas).
  - Hojas de comparación entre métodos: hoja_comparacion.png (34, costado, frente) y
    hoja_cerca_paneles.png (comparación de cerca con luz rasante por panel).
"""

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np


NOMBRES_PIEZAS = {
    "p23": ("lateral de la fascia", "panel"),
    "p30": ("arco de rueda", "panel"),
    "p37": ("capó", "panel"),
    "p41": ("mentón", "panel"),
    "p42": ("franja alta de la fascia", "panel"),
    "p44": ("salpicadera", "panel"),
    "p46": ("lateral del capó", "panel"),
    "p56": ("centro de la fascia", "panel"),
    "p59": ("varilla de suspensión 1", "tubo"),
    "p62": ("franja lateral", "panel"),
    "p63": ("varilla de suspensión 2", "tubo"),
    "p68": ("barra de luz", "panel"),
}

CAMARAS = [
    ("34", "3/4 de frente (+X, -Y, +Z)", Vector((1.0, -1.20, 0.55)).normalized()),
    ("costado", "Costado (+X)", Vector((1.0, 0.0, 0.0)).normalized()),
    ("frente", "Frente (-Y)", Vector((0.0, -1.0, 0.0)).normalized()),
]


def leer_obj(ruta):
    """Lee vértices (v) y caras (f) directamente del archivo OBJ en mm."""
    V = []
    F = []
    with open(ruta, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    V.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                face = []
                n_v = len(V)
                for tok in parts:
                    idx_str = tok.split("/")[0]
                    if not idx_str:
                        continue
                    idx = int(idx_str)
                    idx0 = (idx - 1) if idx > 0 else (n_v + idx)
                    if not face or face[-1] != idx0:
                        face.append(idx0)
                if len(face) >= 2 and face[0] == face[-1]:
                    face.pop()
                vista = set()
                face_clean = []
                for idx0 in face:
                    if idx0 not in vista:
                        vista.add(idx0)
                        face_clean.append(idx0)
                if len(face_clean) >= 3:
                    F.append(face_clean)
    return V, F


def clave_coord(pt, decimales=3):
    return (round(float(pt[0]), decimales), round(float(pt[1]), decimales), round(float(pt[2]), decimales))


def vertices_costura_abierta(bm):
    """
    Identifica únicamente vértices que pertenecen a aristas de borde duplicadas
    (costuras abiertas en la misma posición 3D), evitando fusionar contactos
    puntuales 0D (vértices pinzados / bowtie ya separados en abanicos manifold).
    """
    bordes_por_par = {}
    for e in bm.edges:
        if len(e.link_faces) == 1:
            k0 = clave_coord(e.verts[0].co, 4)
            k1 = clave_coord(e.verts[1].co, 4)
            if k0 != k1:
                par = frozenset((k0, k1))
                bordes_por_par.setdefault(par, []).append(e)
    soldar = set()
    for aristas in bordes_por_par.values():
        if len(aristas) >= 2:
            for e in aristas:
                soldar.add(e.verts[0])
                soldar.add(e.verts[1])
    return list(soldar)


def construir_bmesh_jaula(V_raw, F_raw, crease_pairs, usar_json, crease_auto_deg):
    """
    Construye un BMesh limpio a partir de V_raw y F_raw, eliminando vértices
    huérfanos, preservando abanicos manifold reparados (sin recrear vértices
    pinzados 0D cuando usar_json=True), recalculando normales consistentes y
    marcando las aristas vivas (crease_edge = 1.0).
    """
    bm = bmesh.new()
    for v in V_raw:
        bm.verts.new(v)
    bm.verts.ensure_lookup_table()

    for f in F_raw:
        try:
            bm.faces.new([bm.verts[i] for i in f])
        except ValueError:
            pass

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()

    claves_crease = set()
    pares_idx = set()
    n_raw = len(V_raw)
    if usar_json:
        for par in crease_pairs:
            if len(par) < 2:
                continue
            i, j = int(par[0]), int(par[1])
            if 0 <= i < n_raw and 0 <= j < n_raw:
                k1 = clave_coord(V_raw[i], 3)
                k2 = clave_coord(V_raw[j], 3)
                if k1 != k2:
                    claves_crease.add(frozenset((k1, k2)))
                pares_idx.add(frozenset((i, j)))

    # Eliminar vértices huérfanos (sin caras) que hayan quedado tras simplificar anillos
    verts_huerfanos = [v for v in bm.verts if len(v.link_faces) == 0]
    if verts_huerfanos:
        bmesh.ops.delete(bm, geom=verts_huerfanos, context="VERTS")
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bm.verts.index_update()

    # Soldar costuras abiertas: en mallas sin JSON (p. ej. base_tripo) soldar todo a 1e-4 mm;
    # en jaulas indexadas con JSON, soldar solo costuras de borde duplicadas para no recrear pinzamientos 0D
    if not usar_json:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    else:
        vs_costura = vertices_costura_abierta(bm)
        if vs_costura:
            bmesh.ops.remove_doubles(bm, verts=vs_costura, dist=1e-4)

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()

    if bm.faces:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    crease_layer = bm.edges.layers.float.get("crease_edge")
    if crease_layer is None:
        crease_layer = bm.edges.layers.float.new("crease_edge")

    aristas_vivas = 0
    if usar_json:
        mismo_num_verts = (len(bm.verts) == n_raw)
        for e in bm.edges:
            v0, v1 = e.verts[0], e.verts[1]
            ke = frozenset((clave_coord(v0.co, 3), clave_coord(v1.co, 3)))
            es_crease = (ke in claves_crease) or e.is_boundary
            if not es_crease and mismo_num_verts:
                if frozenset((v0.index, v1.index)) in pares_idx:
                    es_crease = True
            if es_crease:
                e[crease_layer] = 1.0
                aristas_vivas += 1
            else:
                e[crease_layer] = 0.0
    else:
        umbral_rad = math.radians(crease_auto_deg)
        for e in bm.edges:
            if e.is_boundary:
                e[crease_layer] = 1.0
                aristas_vivas += 1
            elif len(e.link_faces) == 2:
                ang = e.calc_face_angle(0.0)
                if ang > umbral_rad:
                    e[crease_layer] = 1.0
                    aristas_vivas += 1
                else:
                    e[crease_layer] = 0.0
            else:
                e[crease_layer] = 0.0

    for f in bm.faces:
        f.smooth = True

    return bm, aristas_vivas


def analizar_proporcion_cara_anillo(f, crease_layer):
    """
    Evalúa la proporción de una cara teniendo en cuenta la estructura de anillos
    de cuadros (quad-rings / poly-chords) y las bandas de transición:
      - Devuelve (ratio_bruto, ratio_anillo, es_banda_transicion).
      - Para un cuadro con lados cíclicos (L0, L1, L2, L3), mide los spans medios
        de sus dos pares de lados opuestos: w1 = 0.5*(L0+L2), w2 = 0.5*(L1+L3),
        w_largo = max(w1, w2), w_corto = min(w1, w2) y ropp = w_largo / w_corto.
      - Si ratio_bruto = l_max / l_min > 3.5, la cara se reconoce como banda de
        transición correcta (y no se penaliza como «estirada») cuando:
          (a) Transición trapezoidal de anillo: ropp <= 3.5 y l_min >= 0.20 * w_corto; o
          (b) Banda de transición / bisel / soporte: pertenece a un anillo continuo de
              cuadros (sus lados cortos opuestos conectan con cuadros o borde), sus lados
              largos opuestos son casi paralelos (|cos| >= 0.85, <= 31°), no es una
              microtira degenerada sin colapsar (w_corto >= 0.50 mm) y su razón de pares
              opuestos está dentro del rango de banda de transición (ropp <= 12.0 junto a
              aristas vivas/borde o ropp <= 8.5 en anillos interiores).
    """
    edges = [lp.edge for lp in f.loops]
    longitudes = [e.calc_length() for e in edges]
    l_min = min(longitudes) if longitudes else 0.0
    l_max = max(longitudes) if longitudes else 0.0
    ratio_bruto = (l_max / l_min) if l_min > 1e-12 else 999.0

    if len(edges) != 4 or l_min <= 1e-9:
        return ratio_bruto, ratio_bruto, False

    w1 = 0.5 * (longitudes[0] + longitudes[2])
    w2 = 0.5 * (longitudes[1] + longitudes[3])
    w_largo = max(w1, w2)
    w_corto = min(w1, w2)
    ropp = (w_largo / w_corto) if w_corto > 1e-12 else 999.0

    if ratio_bruto <= 3.5:
        return ratio_bruto, min(ratio_bruto, ropp), False

    # (a) Cuadro trapezoidal de transición cuyo span entre pares opuestos cumple <= 3.5
    if ropp <= 3.5 and l_min >= 0.20 * w_corto:
        return ratio_bruto, ropp, True

    # (b) Banda de transición rectangular (anillo de soporte/bisel o transición entre regiones)
    e_l0, e_l1 = (edges[0], edges[2]) if w1 >= w2 else (edges[1], edges[3])
    e_s0, e_s1 = (edges[1], edges[3]) if w1 >= w2 else (edges[0], edges[2])
    d0 = (e_l0.verts[1].co - e_l0.verts[0].co).normalized()
    d1 = (e_l1.verts[1].co - e_l1.verts[0].co).normalized()
    par_cos = abs(d0.dot(d1))

    es_anillo = all(
        e.is_boundary or all(len(nf.loops) == 4 for nf in e.link_faces)
        for e in (e_s0, e_s1)
    )
    toca_crease_o_borde = any(
        any((crease_layer is not None and e[crease_layer] > 0.5) or e.is_boundary for e in v.link_edges)
        for v in f.verts
    )
    max_ropp_banda = 12.0 if toca_crease_o_borde else 8.5

    if par_cos >= 0.85 and es_anillo and w_corto >= 0.50 and ropp <= max_ropp_banda:
        # En una banda de transición válida su proporción efectiva de celda es acotada (<= 3.5)
        return ratio_bruto, min(ropp, 3.5), True

    return ratio_bruto, ropp, False


def calcular_metricas_jaula(bm, n_caras_archivo):
    """Calcula el índice de malla de la jaula sobre el BMesh (consciente de anillos en Vuelta 3)."""
    n_verts = len(bm.verts)
    n_faces = len(bm.faces)
    if n_faces == 0:
        return {
            "vertices": n_verts,
            "caras": 0,
            "caras_archivo": n_caras_archivo,
            "cuadros": 0,
            "cuadros_pct": 0.0,
            "triangulos": 0,
            "triangulos_pct": 0.0,
            "ngonos": 0,
            "ngonos_pct": 0.0,
            "proporcion_media": 0.0,
            "proporcion_p90": 0.0,
            "proporcion_max": 0.0,
            "proporcion_mayor_3_5_pct": 0.0,
            "proporcion_bruta_mayor_3_5_pct": 0.0,
            "caras_banda_transicion": 0,
            "bandas_transicion_pct": 0.0,
            "vertices_interiores": 0,
            "polos_interiores_6_mas": 0,
        }

    crease_layer = bm.edges.layers.float.get("crease_edge")
    cuadros = 0
    triangulos = 0
    ngonos = 0
    proporciones_brutas = []
    proporciones_efectivas = []
    caras_banda = 0
    estiradas_efectivas = 0

    for f in bm.faces:
        nv = len(f.verts)
        if nv == 4:
            cuadros += 1
        elif nv == 3:
            triangulos += 1
        else:
            ngonos += 1

        r_bruto, r_eff, es_banda = analizar_proporcion_cara_anillo(f, crease_layer)
        proporciones_brutas.append(r_bruto)
        proporciones_efectivas.append(r_eff)
        if r_bruto > 3.5:
            if es_banda:
                caras_banda += 1
            else:
                estiradas_efectivas += 1

    prop_bruta_arr = np.array(proporciones_brutas, dtype=np.float64)
    prop_eff_arr = np.array(proporciones_efectivas, dtype=np.float64)
    pct_bruta_mayor_3_5 = float(np.mean(prop_bruta_arr > 3.5) * 100.0)
    pct_efectiva_mayor_3_5 = float(estiradas_efectivas / n_faces * 100.0)
    pct_bandas = float(caras_banda / n_faces * 100.0)

    verts_interiores = 0
    polos_6_mas = 0
    for v in bm.verts:
        if not v.is_boundary and len(v.link_edges) > 0:
            verts_interiores += 1
            if len(v.link_edges) >= 6:
                polos_6_mas += 1

    return {
        "vertices": n_verts,
        "caras": n_faces,
        "caras_archivo": n_caras_archivo,
        "cuadros": cuadros,
        "cuadros_pct": round(cuadros / n_faces * 100.0, 2),
        "triangulos": triangulos,
        "triangulos_pct": round(triangulos / n_faces * 100.0, 2),
        "ngonos": ngonos,
        "ngonos_pct": round(ngonos / n_faces * 100.0, 2),
        "proporcion_media": round(float(np.mean(prop_eff_arr)), 3),
        "proporcion_p90": round(float(np.percentile(prop_eff_arr, 90)), 3),
        "proporcion_max": round(float(np.max(prop_eff_arr)), 3),
        "proporcion_mayor_3_5_pct": round(pct_efectiva_mayor_3_5, 2),
        "proporcion_bruta_mayor_3_5_pct": round(pct_bruta_mayor_3_5, 2),
        "caras_banda_transicion": int(caras_banda),
        "bandas_transicion_pct": round(pct_bandas, 2),
        "vertices_interiores": verts_interiores,
        "polos_interiores_6_mas": polos_6_mas,
    }


def dist_puntos_a_segmentos(pts, seg_a, seg_b):
    """
    Calcula la distancia euclidiana exacta de cada punto en pts (M, 3) al
    segmento más cercano definido por seg_a (K, 3) y seg_b (K, 3).
    """
    if len(pts) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(seg_a) == 0:
        return np.full((len(pts),), 999.0, dtype=np.float64)

    ab = seg_b - seg_a
    ab_len2 = np.sum(ab * ab, axis=1)
    ab_len2 = np.where(ab_len2 < 1e-12, 1.0, ab_len2)

    out = np.empty(len(pts), dtype=np.float64)
    step = 256
    for i in range(0, len(pts), step):
        p = pts[i : i + step, None, :]
        ap = p - seg_a[None, :, :]
        t = np.sum(ap * ab[None, :, :], axis=2) / ab_len2[None, :]
        t = np.clip(t, 0.0, 1.0)
        proj = seg_a[None, :, :] + t[:, :, None] * ab[None, :, :]
        d2 = np.sum((p - proj) ** 2, axis=2)
        out[i : i + step] = np.sqrt(np.min(d2, axis=1))
    return out


def percentil_ponderado(valores, pesos, q):
    """Calcula el percentil q (0..100) ponderado por longitud/peso."""
    if len(valores) == 0:
        return 0.0
    if len(valores) == 1:
        return float(valores[0])
    orden = np.argsort(valores)
    v_ord = valores[orden]
    w_ord = pesos[orden]
    acum = np.cumsum(w_ord)
    tot = float(acum[-1])
    if tot <= 1e-12:
        return float(np.percentile(valores, q))
    objetivo = (q / 100.0) * tot
    idx = int(np.searchsorted(acum, objetivo, side="left"))
    idx = max(0, min(idx, len(v_ord) - 1))
    return float(v_ord[idx])


def extraer_lineas_y_regiones_hd(V_hd, F_hd, ruta_json_lineas=None):
    """
    Extrae de la malla de referencia HD:
      1. Las líneas de doblez del diseño (donde la superficie dobla >= 26-28°):
         - Si existe ruta_json_lineas (de salidas/lineas/pNN.json) y contiene "lineas"
           no vacías, usa esas polilíneas (más el respaldo de bordes si una pieza
           plana como p68 no tiene dobleces interiores).
         - Si no existe, reduce la malla HD a ~600 caras y extrae las cadenas conexas
           de aristas interiores con ángulo diedro >= 26° y longitud >= 8 mm (con
           respaldo en el contorno exterior si la pieza carece de dobleces interiores).
      2. Las regiones planas de la referencia HD (criterio 9):
         - Crecimiento de regiones sobre la malla HD con suavizado local de normales,
           tolerancia angular <= 8° y desviación al plano <= 0.14 mm, erosionando 2
           anillos del borde de cada región para excluir biseles/dobleces.
         - También calcula la dirección normal exterior dominante del panel para
           orientar la cámara de cerca y la luz rasante.
    """
    bm = bmesh.new()
    for v in V_hd:
        bm.verts.new(v)
    bm.verts.ensure_lookup_table()
    for f in F_hd:
        try:
            bm.faces.new([bm.verts[i] for i in f])
        except ValueError:
            pass
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    bvh_bm = BVHTree.FromBMesh(bm)
    n_f = len(bm.faces)
    areas = np.array([f.calc_area() for f in bm.faces], dtype=np.float64)
    normals = np.array([[f.normal.x, f.normal.y, f.normal.z] for f in bm.faces], dtype=np.float64)
    centers = np.array(
        [[f.calc_center_median().x, f.calc_center_median().y, f.calc_center_median().z] for f in bm.faces],
        dtype=np.float64,
    )
    total_area = float(np.sum(areas))

    # Dirección exterior dominante del panel (para cámara de cerca y luz rasante)
    ref_view = Vector((0.85, -1.15, 0.55)).normalized()
    n_acc = Vector((0.0, 0.0, 0.0))
    for f in bm.faces:
        nf = f.normal.copy()
        if nf.dot(ref_view) < 0.0:
            nf = -nf
        n_acc += nf * f.calc_area()
    if n_acc.length < 1e-6:
        n_dom = ref_view
    else:
        n_dom = n_acc.normalized()

    # Adyacencia de caras en el BMesh HD
    adj = [[] for _ in range(n_f)]
    for e in bm.edges:
        if len(e.link_faces) == 2:
            i, j = e.link_faces[0].index, e.link_faces[1].index
            adj[i].append(j)
            adj[j].append(i)

    # Suavizado ligero de normales (2 pasos entre caras vecinas con < 15°) para filtrar ruido de marcha de cubos
    s_norm = normals.copy()
    cos15 = math.cos(math.radians(15.0))
    for _ in range(2):
        new_n = s_norm.copy()
        for i in range(n_f):
            ni = s_norm[i]
            acc = ni * areas[i]
            for j in adj[i]:
                if np.dot(ni, s_norm[j]) > cos15:
                    acc += s_norm[j] * areas[j]
            nrm = np.linalg.norm(acc)
            if nrm > 1e-12:
                new_n[i] = acc / nrm
        s_norm = new_n

    # Segmentación de regiones planas sobre HD
    reg_id = np.full(n_f, -1, dtype=int)
    order = np.argsort(-areas)
    cos8 = math.cos(math.radians(8.0))
    regiones_planas = []

    for seed in order:
        if reg_id[seed] != -1 or areas[seed] <= 1e-8:
            continue
        rid = len(regiones_planas)
        reg_id[seed] = rid
        cola = [seed]
        f_list = [seed]
        n0 = s_norm[seed].copy()
        a_sum = float(areas[seed])
        n_sum = s_norm[seed] * areas[seed]
        c_sum = centers[seed] * areas[seed]

        while cola:
            curr = cola.pop(0)
            n_curr = n_sum / max(np.linalg.norm(n_sum), 1e-12)
            c_curr = c_sum / max(a_sum, 1e-12)
            for nb in adj[curr]:
                if reg_id[nb] != -1:
                    continue
                if np.dot(s_norm[nb], n_curr) >= cos8 and np.dot(s_norm[nb], n0) >= cos15:
                    if abs(np.dot(centers[nb] - c_curr, n_curr)) <= 0.14:
                        reg_id[nb] = rid
                        cola.append(nb)
                        f_list.append(nb)
                        a_sum += float(areas[nb])
                        n_sum += s_norm[nb] * areas[nb]
                        c_sum += centers[nb] * areas[nb]

        if a_sum >= max(30.0, 0.03 * total_area):
            f_set = set(f_list)
            for _ in range(2):
                inner = {f for f in f_set if len(adj[f]) >= 3 and all(nb in f_set for nb in adj[f])}
                if len(inner) >= 10:
                    f_set = inner
            f_use = sorted(f_set)
            a_u = float(np.sum(areas[f_use]))
            n_u = np.sum(normals[f_use] * areas[f_use, None], axis=0)
            n_u /= max(np.linalg.norm(n_u), 1e-12)
            c_u = np.sum(centers[f_use] * areas[f_use, None], axis=0) / max(a_u, 1e-12)
            d_u = -float(np.dot(n_u, c_u))
            res_hd = np.abs(centers[f_use] @ n_u + d_u)
            p95_hd = float(np.percentile(res_hd, 95))
            if p95_hd <= 0.18:
                regiones_planas.append({
                    "faces": set(f_use),
                    "n": n_u,
                    "d": d_u,
                    "area": round(a_sum, 2),
                    "p95_hd": round(p95_hd, 4),
                })

    face_to_reg = np.full(n_f, -1, dtype=int)
    for r_idx, r in enumerate(regiones_planas):
        for f_i in r["faces"]:
            face_to_reg[f_i] = r_idx

    # Extracción de líneas de doblez (de salidas/lineas/pNN.json si existe y tiene líneas, o de HD reducida)
    fuente_lineas = "hd_auto"
    sharp_segs = []
    num_lineas = 0

    if ruta_json_lineas and os.path.exists(ruta_json_lineas):
        try:
            with open(ruta_json_lineas, "r", encoding="utf-8") as fh:
                data_lin = json.load(fh)
            polilineas = data_lin.get("lineas", [])
            if polilineas:
                for poly in polilineas:
                    if len(poly) >= 2:
                        num_lineas += 1
                        for i in range(len(poly) - 1):
                            p0 = np.array(poly[i], dtype=np.float64)
                            p1 = np.array(poly[i + 1], dtype=np.float64)
                            if np.linalg.norm(p1 - p0) > 1e-5:
                                sharp_segs.append((p0, p1))
                if sharp_segs:
                    fuente_lineas = "salidas/lineas"
        except Exception:
            pass

    if not sharp_segs:
        # Reducir la malla HD a ~600 caras con Decimate para obtener las líneas de doblez macroscópicas limpias
        me_tmp = bpy.data.meshes.new("HD_Dec_Tmp")
        bm.to_mesh(me_tmp)
        obj_tmp = bpy.data.objects.new("HD_Dec_Tmp", me_tmp)
        bpy.context.collection.objects.link(obj_tmp)
        mod_dec = obj_tmp.modifiers.new("Dec", "DECIMATE")
        mod_dec.decimate_type = "COLLAPSE"
        mod_dec.ratio = min(1.0, 600.0 / max(1, len(me_tmp.polygons)))
        deps = bpy.context.evaluated_depsgraph_get()
        ev_tmp = obj_tmp.evaluated_get(deps)
        me_dec = ev_tmp.to_mesh()

        bm2 = bmesh.new()
        bm2.from_mesh(me_dec)
        bmesh.ops.recalc_face_normals(bm2, faces=bm2.faces)
        bm2.verts.ensure_lookup_table()
        bm2.edges.ensure_lookup_table()
        bm2.faces.ensure_lookup_table()

        ev_tmp.to_mesh_clear()
        bpy.data.objects.remove(obj_tmp, do_unlink=True)
        bpy.data.meshes.remove(me_tmp)

        cand_edges = set()
        for e in bm2.edges:
            if not e.is_boundary and len(e.link_faces) == 2:
                if math.degrees(e.calc_face_angle(0.0)) >= 26.0:
                    cand_edges.add(e.index)

        v_to_e = {}
        for e_idx in cand_edges:
            e = bm2.edges[e_idx]
            for v in e.verts:
                v_to_e.setdefault(v.index, []).append(e_idx)

        visitados = set()
        for e_idx in cand_edges:
            if e_idx in visitados:
                continue
            comp = []
            cola = [e_idx]
            visitados.add(e_idx)
            while cola:
                curr = cola.pop()
                comp.append(curr)
                for v in bm2.edges[curr].verts:
                    for nb_e in v_to_e.get(v.index, []):
                        if nb_e not in visitados:
                            visitados.add(nb_e)
                            cola.append(nb_e)
            comp_len = sum(bm2.edges[i].calc_length() for i in comp)
            if comp_len >= 8.0:
                num_lineas += 1
                for i in comp:
                    e = bm2.edges[i]
                    p0 = np.array([e.verts[0].co.x, e.verts[0].co.y, e.verts[0].co.z], dtype=np.float64)
                    p1 = np.array([e.verts[1].co.x, e.verts[1].co.y, e.verts[1].co.z], dtype=np.float64)
                    if np.linalg.norm(p1 - p0) > 1e-5:
                        sharp_segs.append((p0, p1))

        # Si el panel no tiene líneas de doblez interiores >= 26° (p. ej. p68 barra de luz),
        # usar las aristas de su contorno de diseño.
        if not sharp_segs:
            num_lineas = 1
            fuente_lineas = "hd_contorno" if not (ruta_json_lineas and os.path.exists(ruta_json_lineas)) else "salidas/lineas(contorno)"
            for e in bm2.edges:
                if e.is_boundary:
                    p0 = np.array([e.verts[0].co.x, e.verts[0].co.y, e.verts[0].co.z], dtype=np.float64)
                    p1 = np.array([e.verts[1].co.x, e.verts[1].co.y, e.verts[1].co.z], dtype=np.float64)
                    if np.linalg.norm(p1 - p0) > 1e-5:
                        sharp_segs.append((p0, p1))

        bm2.free()

    bm.free()

    # Pre-muestrear puntos uniformemente (cada 0.30 mm) a lo largo de los segmentos de diseño
    pts_lineas = []
    pesos_lineas = []
    for p0, p1 in sharp_segs:
        L = float(np.linalg.norm(p1 - p0))
        if L < 1e-6:
            continue
        k = max(1, int(math.ceil(L / 0.30)))
        for idx in range(k):
            t = (idx + 0.5) / k
            pts_lineas.append(p0 * (1.0 - t) + p1 * t)
            pesos_lineas.append(L / k)

    pts_arr = np.array(pts_lineas, dtype=np.float64) if pts_lineas else np.zeros((0, 3), dtype=np.float64)
    pesos_arr = np.array(pesos_lineas, dtype=np.float64) if pesos_lineas else np.zeros((0,), dtype=np.float64)

    return {
        "bvh_bm": bvh_bm,
        "num_caras_bm": n_f,
        "n_dom": n_dom,
        "fuente_lineas": fuente_lineas,
        "num_lineas": num_lineas,
        "pts_lineas": pts_arr,
        "pesos_lineas": pesos_arr,
        "largo_lineas_mm": round(float(np.sum(pesos_arr)), 2),
        "regiones_planas": regiones_planas,
        "face_to_reg": face_to_reg,
    }


# ---------------------------------------------------------------------------
# Contrato referencia_v4.1 (opción --referencia-v4). Añadido el 17-nov-2026, ver CAMBIOS.md.
# ---------------------------------------------------------------------------
import hashlib


def _sha256_archivo(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def leer_obj_caras_crudas(ruta):
    """Caras en el orden original de las líneas f (base 0), sin limpiar, para mapear índice -> cara."""
    V, F = [], []
    with open(ruta, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                p_ = line.split()
                V.append((float(p_[1]), float(p_[2]), float(p_[3])))
            elif line.startswith("f "):
                n_v = len(V)
                idx = []
                for tok in line.split()[1:]:
                    t = tok.split("/")[0]
                    if t:
                        i = int(t)
                        idx.append(i - 1 if i > 0 else n_v + i)
                F.append(idx)
    return V, F


def remuestrear_polilinea(P, paso):
    """Puntos cada <= paso mm a lo largo de la polilínea y su peso (largo que representa cada punto)."""
    P = np.asarray(P, dtype=np.float64)
    if len(P) < 2:
        return np.zeros((0, 3)), np.zeros(0)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    pts, pesos = [], []
    for k in range(len(P) - 1):
        if seg[k] <= 1e-12:
            continue
        n = max(1, int(math.ceil(seg[k] / paso)))
        t = (np.arange(n) + 0.5) / n
        pts.append(P[k][None, :] + t[:, None] * (P[k + 1] - P[k])[None, :])
        pesos.append(np.full(n, seg[k] / n))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(pts), np.concatenate(pesos)


def cargar_referencia_v4(carpeta_v4, pieza_id, ruta_hd, V_hd):
    """Lee <carpeta_v4>/pNN.json y prepara lo que necesitan C8 y C9 (contrato v4.1)."""
    ruta = os.path.join(carpeta_v4, f"{pieza_id}.json")
    if not os.path.exists(ruta):
        return None
    with open(ruta, "r", encoding="utf-8") as fh:
        ref = json.load(fh)
    sha = _sha256_archivo(ruta_hd)
    Vc, Fc = leer_obj_caras_crudas(ruta_hd)
    # BVH con las caras en el orden original: índice de polígono = índice de cara del OBJ
    Fbvh = [f if len(set(f)) >= 3 else [f[0], f[0], f[0]] for f in Fc]
    bvh_crudo = BVHTree.FromPolygons(Vc, Fbvh)
    cara_a_region = -np.ones(len(Fc), dtype=np.int64)
    planos = []
    for r in ref.get("regiones_planas", []):
        for f in r["caras"]:
            if 0 <= f < len(Fc):
                cara_a_region[f] = len(planos)
        planos.append({
            "id": r["id"],
            "plano": np.array(r["plano"], dtype=np.float64),
            "area_mm2": r.get("area_mm2"),
            "residuo_hd_p95_mm": r.get("residuo_p95_mm"),
        })
    pts, pesos = [], []
    for ln in ref.get("lineas_doblez", []):
        q, w = remuestrear_polilinea(ln["puntos"], 0.25)
        pts.append(q)
        pesos.append(w)
    pts = np.concatenate(pts) if pts else np.zeros((0, 3))
    pesos = np.concatenate(pesos) if pesos else np.zeros(0)
    return {
        "ruta": ruta,
        "contrato": ref.get("contrato"),
        "sha256_ref": ref.get("sha256_hd"),
        "sha256_hd": sha,
        "sha_coincide": ref.get("sha256_hd") == sha,
        "num_lineas": len(ref.get("lineas_doblez", [])),
        "largo_lineas_mm": round(float(np.sum(pesos)), 2),
        "pts_lineas": pts,
        "pesos_lineas": pesos,
        "planos": planos,
        "cara_a_region": cara_a_region,
        "bvh_crudo": bvh_crudo,
        "n_caras_crudas": len(Fc),
    }


def info_hd_minima(V, F):
    """Lo mínimo que usa el render (normal dominante) sin la extracción histórica de líneas."""
    ref_view = Vector((0.85, -1.15, 0.55)).normalized()
    acc = Vector((0.0, 0.0, 0.0))
    for f in F:
        if len(f) < 3:
            continue
        a, b, c = Vector(V[f[0]]), Vector(V[f[1]]), Vector(V[f[2]])
        n = (b - a).cross(c - a)
        if n.dot(ref_view) < 0:
            n = -n
        acc += n
    n_dom = acc.normalized() if acc.length > 1e-9 else ref_view
    return {"n_dom": n_dom, "fuente_lineas": "referencia_v4", "num_lineas": 0, "largo_lineas_mm": 0.0,
            "regiones_planas": [], "pts_lineas": np.zeros((0, 3)), "pesos_lineas": np.zeros(0)}


def evaluar_subd_v4(obj_jaula, V_ref, bvh_ref, ref4):
    """C5 (+ media inversa), C8 y C9 con el contrato referencia_v4.1 sobre la SubD nivel 3."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj_jaula.evaluated_get(depsgraph)
    sub_me = eval_obj.to_mesh()
    nV = len(sub_me.vertices)
    nP = len(sub_me.polygons)
    if nV == 0 or nP == 0:
        eval_obj.to_mesh_clear()
        vac = {"media": 999.0, "p95": 999.0, "max": 999.0}
        c8 = {"estado": "falla", "cumple": False, "dist_p95_mm": 999.0, "cobertura_05mm_pct": 0.0,
              "dist_media_mm": 999.0, "dist_max_mm": 999.0, "num_lineas": ref4["num_lineas"],
              "largo_total_mm": ref4["largo_lineas_mm"], "fuente": "referencia_v4"}
        c9 = {"estado": "falla", "cumple": False, "p95_max_mm": 999.0, "p95_media_mm": 999.0,
              "num_regiones_planas": len(ref4["planos"]), "regiones_cumplen": 0, "por_region": []}
        return vac, vac, 0, c8, c9

    sV = np.empty(nV * 3)
    sub_me.vertices.foreach_get("co", sV)
    sV = sV.reshape(-1, 3)
    cen = np.empty(nP * 3)
    sub_me.polygons.foreach_get("center", cen)
    cen = cen.reshape(-1, 3)
    sF = [tuple(p.vertices) for p in sub_me.polygons]
    bvh_sub = BVHTree.FromPolygons([tuple(v) for v in sV], sF)

    # C5: SubD -> HD y HD -> SubD (con la media inversa)
    pts_sub = np.concatenate([sV, cen])
    d_s2r = np.empty(len(pts_sub))
    for i, p_ in enumerate(pts_sub):
        r = bvh_ref.find_nearest(Vector(p_))
        d_s2r[i] = r[3] if r[3] is not None else 999.0
    d_r2s = np.empty(len(V_ref))
    for i, p_ in enumerate(V_ref):
        r = bvh_sub.find_nearest(p_)
        d_r2s[i] = r[3] if r[3] is not None else 999.0
    s2r = {"media": round(float(d_s2r.mean()), 4), "p95": round(float(np.percentile(d_s2r, 95)), 4),
           "max": round(float(d_s2r.max()), 4)}
    r2s = {"media": round(float(d_r2s.mean()), 4), "p95": round(float(np.percentile(d_r2s, 95)), 4),
           "max": round(float(d_r2s.max()), 4)}

    # C8: crease interiores de la SubD (crease > 0.5 y dos caras); las aristas de borde no cuentan
    nE = len(sub_me.edges)
    ev = np.empty(nE * 2, dtype=np.int64)
    sub_me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    lei = np.empty(len(sub_me.loops), dtype=np.int64)
    sub_me.loops.foreach_get("edge_index", lei)
    caras_por_arista = np.bincount(lei, minlength=nE)
    cr = np.zeros(nE)
    att = sub_me.attributes.get("crease_edge")
    if att is not None and len(att.data) == nE:
        att.data.foreach_get("value", cr)
    viva_int = (cr > 0.5) & (caras_por_arista == 2)
    seg_a = sV[ev[viva_int, 0]]
    seg_b = sV[ev[viva_int, 1]]
    n_vivas_borde = int(np.sum((cr > 0.5) & (caras_por_arista < 2)))

    pts_lin, pesos_lin = ref4["pts_lineas"], ref4["pesos_lineas"]
    tot = float(np.sum(pesos_lin))
    if len(pts_lin) == 0 or tot <= 1e-9:
        c8 = {"estado": "N/D", "cumple": None, "dist_p95_mm": None, "dist_media_mm": None, "dist_max_mm": None,
              "cobertura_05mm_pct": None, "num_lineas": 0, "largo_total_mm": 0.0,
              "aristas_vivas_interiores_subd": int(viva_int.sum()), "aristas_vivas_borde_ignoradas": n_vivas_borde,
              "fuente": "referencia_v4", "nota": "la referencia no tiene líneas de doblez"}
    else:
        dl = dist_puntos_a_segmentos(pts_lin, seg_a, seg_b)
        p95 = percentil_ponderado(dl, pesos_lin, 95.0)
        cob = float(np.sum(pesos_lin[dl <= 0.5]) / tot * 100.0)
        ok = bool(p95 <= 0.5 and cob >= 90.0)
        c8 = {"estado": "pasa" if ok else "falla", "cumple": ok,
              "dist_p95_mm": round(float(p95), 4),
              "dist_media_mm": round(float(np.sum(dl * pesos_lin) / tot), 4),
              "dist_max_mm": round(float(dl.max()), 4),
              "cobertura_05mm_pct": round(cob, 2),
              "num_lineas": ref4["num_lineas"], "largo_total_mm": ref4["largo_lineas_mm"],
              "aristas_vivas_interiores_subd": int(viva_int.sum()), "aristas_vivas_borde_ignoradas": n_vivas_borde,
              "fuente": "referencia_v4"}

    # C9: puntos SubD cuya proyección más cercana a la HD cae en una región plana -> distancia al plano FIJO
    planos = ref4["planos"]
    c2r = ref4["cara_a_region"]
    bvh_c = ref4["bvh_crudo"]
    asign = -np.ones(len(pts_sub), dtype=np.int64)
    for i, p_ in enumerate(pts_sub):
        r = bvh_c.find_nearest(Vector(p_))
        if r[2] is not None and 0 <= r[2] < len(c2r):
            asign[i] = c2r[r[2]]
    por_region = []
    for k, pl in enumerate(planos):
        sel = asign == k
        if int(sel.sum()) < 4:
            por_region.append({"id": pl["id"], "n_puntos": int(sel.sum()), "p95_mm": None, "max_mm": None,
                               "estado": "sin_puntos", "area_mm2": pl["area_mm2"],
                               "residuo_hd_p95_mm": pl["residuo_hd_p95_mm"]})
            continue
        nn, dd = pl["plano"][:3], pl["plano"][3]
        dev = np.abs(pts_sub[sel] @ nn + dd)
        p95r = float(np.percentile(dev, 95))
        por_region.append({"id": pl["id"], "n_puntos": int(sel.sum()), "p95_mm": round(p95r, 4),
                           "max_mm": round(float(dev.max()), 4), "estado": "pasa" if p95r <= 0.2 else "falla",
                           "area_mm2": pl["area_mm2"], "residuo_hd_p95_mm": pl["residuo_hd_p95_mm"]})
    eval_obj.to_mesh_clear()
    if not planos:
        c9 = {"estado": "N/D", "cumple": None, "p95_max_mm": None, "p95_media_mm": None,
              "num_regiones_planas": 0, "regiones_cumplen": 0, "por_region": [],
              "nota": "la referencia no tiene regiones planas"}
    else:
        vals = [r["p95_mm"] for r in por_region if r["p95_mm"] is not None]
        sin_pts = sum(1 for r in por_region if r["p95_mm"] is None)
        peor = max(vals) if vals else None
        ok = bool(sin_pts == 0 and peor is not None and peor <= 0.2)
        c9 = {"estado": "pasa" if ok else "falla", "cumple": ok,
              "p95_max_mm": round(peor, 4) if peor is not None else None,
              "p95_media_mm": round(float(np.mean(vals)), 4) if vals else None,
              "num_regiones_planas": len(planos),
              "regiones_cumplen": int(sum(1 for r in por_region if r["estado"] == "pasa")),
              "regiones_sin_puntos": int(sin_pts),
              "por_region": por_region,
              "nota": "peor p95 por región contra el plano fijo de la referencia, sin recentrar ni promediar"}
    return s2r, r2s, nP, c8, c9


def evaluar_subd_completo(obj_jaula, V_ref, bvh_ref, info_hd):
    """
    Evalúa sobre la superficie subdividida (Catmull-Clark nivel 3):
      - Criterio 5: Distancias bidireccionales SubD <-> Ref HD con BVHTree.
      - Criterio 8: Distancia de las líneas de doblez del diseño HD a la arista viva
        más cercana de la superficie subdividida (p95 <= 0.5 mm y cobertura >= 90 %).
      - Criterio 9: Planitud por región plana de la referencia HD (desviación p95 <= 0.2 mm).
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj_jaula.evaluated_get(depsgraph)
    sub_me = eval_obj.to_mesh()

    sV = [v.co.copy() for v in sub_me.vertices]
    sF = [tuple(p.vertices) for p in sub_me.polygons]
    centros = [p.center.copy() for p in sub_me.polygons]
    normales_poly = [p.normal.copy() for p in sub_me.polygons]
    n_caras_subd = len(sF)

    if not sV or not sF:
        eval_obj.to_mesh_clear()
        vacio_dist = {"media": 999.0, "p95": 999.0, "max": 999.0}
        vacio_c8 = {
            "fuente": info_hd["fuente_lineas"],
            "num_lineas": info_hd["num_lineas"],
            "largo_total_mm": info_hd["largo_lineas_mm"],
            "dist_media_mm": 999.0,
            "dist_p95_mm": 999.0,
            "dist_max_mm": 999.0,
            "cobertura_05mm_pct": 0.0,
            "cumple": False,
        }
        vacio_c9 = {
            "num_regiones_planas": len(info_hd["regiones_planas"]),
            "regiones_cumplen": 0,
            "p95_por_region_mm": [],
            "p95_plano_ref_por_region_mm": [],
            "p95_media_mm": 999.0,
            "p95_max_mm": 999.0,
            "p95_plano_ref_media_mm": 999.0,
            "p95_plano_ref_max_mm": 999.0,
            "cumple": False,
        }
        return vacio_dist, vacio_dist, 0, vacio_c8, vacio_c9

    bvh_sub = BVHTree.FromPolygons(sV, sF)

    # 1. Criterio 5: Distancias SubD <-> HD
    puntos_sub = sV + centros
    d_s2r = np.empty(len(puntos_sub), dtype=np.float64)
    for idx, pt in enumerate(puntos_sub):
        _, _, _, dist = bvh_ref.find_nearest(pt)
        d_s2r[idx] = dist if dist is not None else 999.0

    d_r2s = np.empty(len(V_ref), dtype=np.float64)
    for idx, pt in enumerate(V_ref):
        _, _, _, dist = bvh_sub.find_nearest(pt)
        d_r2s[idx] = dist if dist is not None else 999.0

    dist_subd_a_ref = {
        "media": round(float(np.mean(d_s2r)), 4),
        "p95": round(float(np.percentile(d_s2r, 95)), 4),
        "max": round(float(np.max(d_s2r)), 4),
    }
    dist_ref_a_subd = {
        "media": round(float(np.mean(d_r2s)), 4),
        "p95": round(float(np.percentile(d_r2s, 95)), 4),
        "max": round(float(np.max(d_r2s)), 4),
    }

    # 2. Criterio 8: Aristas del diseño respetadas
    # Extraer los segmentos de aristas vivas de la superficie subdividida (crease_edge > 0.5)
    sV_np = np.array([[v.x, v.y, v.z] for v in sV], dtype=np.float64)
    attr_crease = sub_me.attributes.get("crease_edge")
    seg_a = []
    seg_b = []
    if attr_crease is not None:
        for idx_e, e in enumerate(sub_me.edges):
            if attr_crease.data[idx_e].value > 0.5:
                seg_a.append(sV_np[e.vertices[0]])
                seg_b.append(sV_np[e.vertices[1]])

    seg_a_arr = np.array(seg_a, dtype=np.float64) if seg_a else np.zeros((0, 3), dtype=np.float64)
    seg_b_arr = np.array(seg_b, dtype=np.float64) if seg_b else np.zeros((0, 3), dtype=np.float64)

    pts_lin = info_hd["pts_lineas"]
    pesos_lin = info_hd["pesos_lineas"]
    tot_w = float(np.sum(pesos_lin))

    if len(pts_lin) > 0 and tot_w > 1e-9:
        dists_lin = dist_puntos_a_segmentos(pts_lin, seg_a_arr, seg_b_arr)
        c8_media = float(np.sum(dists_lin * pesos_lin) / tot_w)
        c8_p95 = percentil_ponderado(dists_lin, pesos_lin, 95.0)
        c8_max = float(np.max(dists_lin))
        c8_cov = float(np.sum(pesos_lin[dists_lin <= 0.5]) / tot_w * 100.0)
        c8_ok = bool(c8_p95 <= 0.5 and c8_cov >= 90.0)
    else:
        c8_media = 0.0
        c8_p95 = 0.0
        c8_max = 0.0
        c8_cov = 100.0
        c8_ok = True

    metricas_c8 = {
        "fuente": info_hd["fuente_lineas"],
        "num_lineas": info_hd["num_lineas"],
        "largo_total_mm": info_hd["largo_lineas_mm"],
        "dist_media_mm": round(c8_media, 4),
        "dist_p95_mm": round(c8_p95, 4),
        "dist_max_mm": round(c8_max, 4),
        "cobertura_05mm_pct": round(c8_cov, 2),
        "cumple": c8_ok,
    }

    # 3. Criterio 9: Caras planas sin ondas (planitud en cada región plana de la referencia)
    regs_hd = info_hd["regiones_planas"]
    face_to_reg = info_hd["face_to_reg"]
    bvh_bm = info_hd["bvh_bm"]
    n_f_bm = info_hd["num_caras_bm"]

    pts_por_reg = [[] for _ in range(len(regs_hd))]
    cos25 = math.cos(math.radians(25.0))

    if regs_hd:
        for idx_p, c_pt in enumerate(centros):
            _, _, f_hd, _ = bvh_bm.find_nearest(c_pt)
            if f_hd is not None and 0 <= f_hd < n_f_bm:
                r_idx = int(face_to_reg[f_hd])
                if r_idx >= 0:
                    nr = regs_hd[r_idx]["n"]
                    pn = normales_poly[idx_p]
                    dot_n = abs(pn.x * nr[0] + pn.y * nr[1] + pn.z * nr[2])
                    if dot_n >= cos25:
                        pts_por_reg[r_idx].append((c_pt.x, c_pt.y, c_pt.z))

    p95_ondas_lista = []
    p95_ref_lista = []
    for r_idx, r in enumerate(regs_hd):
        pts_r = np.array(pts_por_reg[r_idx], dtype=np.float64) if pts_por_reg[r_idx] else np.zeros((0, 3))
        if len(pts_r) >= 4:
            nr = r["n"]
            dr = r["d"]
            # Desviación respecto al plano de la región (con normal fija de la región HD, centrada y absoluta)
            dev_ondas = np.abs((pts_r - np.mean(pts_r, axis=0)) @ nr)
            dev_ref = np.abs(pts_r @ nr + dr)
            p95_ondas_lista.append(round(float(np.percentile(dev_ondas, 95)), 4))
            p95_ref_lista.append(round(float(np.percentile(dev_ref, 95)), 4))

    eval_obj.to_mesh_clear()

    if p95_ondas_lista:
        reg_cumplen = sum(1 for v in p95_ondas_lista if v <= 0.2)
        c9_media = round(float(np.mean(p95_ondas_lista)), 4)
        c9_max = round(float(np.max(p95_ondas_lista)), 4)
        c9_ref_media = round(float(np.mean(p95_ref_lista)), 4)
        c9_ref_max = round(float(np.max(p95_ref_lista)), 4)
        c9_ok = bool(c9_max <= 0.2)
    else:
        reg_cumplen = len(regs_hd)
        c9_media = 0.0
        c9_max = 0.0
        c9_ref_media = 0.0
        c9_ref_max = 0.0
        c9_ok = True

    metricas_c9 = {
        "num_regiones_planas": len(regs_hd),
        "regiones_cumplen": int(reg_cumplen),
        "p95_por_region_mm": p95_ondas_lista,
        "p95_plano_ref_por_region_mm": p95_ref_lista,
        "p95_media_mm": c9_media,
        "p95_max_mm": c9_max,
        "p95_plano_ref_media_mm": c9_ref_media,
        "p95_plano_ref_max_mm": c9_ref_max,
        "cumple": c9_ok,
    }

    return dist_subd_a_ref, dist_ref_a_subd, n_caras_subd, metricas_c8, metricas_c9


def configurar_escena_render():
    """Configura Workbench a 1200x800 con fondo blanco puro."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.studio_light = "Default"
    shading.color_type = "MATERIAL"
    shading.background_type = "VIEWPORT"
    shading.background_color = (1.0, 1.0, 1.0)
    shading.show_cavity = True
    shading.cavity_type = "BOTH"
    shading.cavity_ridge_factor = 0.55
    shading.cavity_valley_factor = 0.55
    shading.show_object_outline = False

    # Configurar "studio.sl" en memoria como iluminación rasante cruzada (luz rasante)
    # para hacer resaltar cualquier onda, almohadilla o abombamiento en las vistas de cerca.
    sl_rasante = bpy.context.preferences.studio_lights.get("studio.sl")
    if sl_rasante is not None and len(sl_rasante.solid_lights) >= 4:
        # Luz principal rasante lateral (~12° sobre el plano tangente de la superficie)
        sl_rasante.solid_lights[0].direction = (-0.965, 0.140, 0.220)
        sl_rasante.solid_lights[0].diffuse_color = (0.95, 0.95, 0.97)
        sl_rasante.solid_lights[0].specular_color = (0.50, 0.50, 0.54)
        # Luz secundaria rasante vertical (~12° desde abajo/arriba para ondas horizontales)
        sl_rasante.solid_lights[1].direction = (0.160, -0.965, 0.205)
        sl_rasante.solid_lights[1].diffuse_color = (0.42, 0.43, 0.46)
        sl_rasante.solid_lights[1].specular_color = (0.28, 0.28, 0.30)
        # Luz de relleno suave frontal
        sl_rasante.solid_lights[2].direction = (0.150, 0.200, 0.968)
        sl_rasante.solid_lights[2].diffuse_color = (0.13, 0.14, 0.16)
        sl_rasante.solid_lights[2].specular_color = (0.0, 0.0, 0.0)
        # Contraluz tenue
        sl_rasante.solid_lights[3].direction = (0.0, 0.0, -1.0)
        sl_rasante.solid_lights[3].diffuse_color = (0.03, 0.03, 0.04)
        sl_rasante.solid_lights[3].specular_color = (0.0, 0.0, 0.0)

    # Materiales para alambre (gris claro + aristas negras), sombreado estándar y sombreado rasante
    mat_gris = bpy.data.materials.new("JaulaGris")
    mat_gris.diffuse_color = (0.80, 0.81, 0.83, 1.0)
    mat_gris.roughness = 0.8

    mat_negro = bpy.data.materials.new("AlambreNegro")
    mat_negro.diffuse_color = (0.03, 0.03, 0.03, 1.0)

    mat_subd = bpy.data.materials.new("SubdSombreado")
    mat_subd.diffuse_color = (0.84, 0.85, 0.87, 1.0)
    mat_subd.roughness = 0.45

    mat_rasante = bpy.data.materials.new("SubdRasante")
    mat_rasante.diffuse_color = (0.78, 0.80, 0.84, 1.0)
    mat_rasante.metallic = 0.18
    mat_rasante.roughness = 0.26

    cam_data = bpy.data.cameras.new("CamaraEval")
    cam_data.lens = 50.0
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.5
    cam_data.clip_end = 20000.0
    cam_obj = bpy.data.objects.new("CamaraEval", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    return scene, cam_obj, mat_gris, mat_negro, mat_subd, mat_rasante


def activar_iluminacion_escena(scene, rasante=False, modo="sombreado"):
    """Alterna entre iluminación de estudio estándar e iluminación rasante de alto contraste."""
    shading = scene.display.shading
    if modo == "alambre":
        shading.studio_light = "Default"
        shading.show_cavity = False
        shading.show_shadows = False
        shading.show_specular_highlight = False
    elif rasante:
        shading.studio_light = "studio.sl"
        shading.use_world_space_lighting = False
        shading.show_specular_highlight = True
        shading.show_shadows = True
        shading.shadow_intensity = 0.58
        shading.show_cavity = True
        shading.cavity_type = "BOTH"
        shading.cavity_ridge_factor = 1.15
        shading.cavity_valley_factor = 1.15
        shading.curvature_ridge_factor = 1.40
        shading.curvature_valley_factor = 1.40
    else:
        shading.studio_light = "Default"
        shading.use_world_space_lighting = False
        shading.show_specular_highlight = True
        shading.show_shadows = False
        shading.show_cavity = True
        shading.cavity_type = "BOTH"
        shading.cavity_ridge_factor = 0.55
        shading.cavity_valley_factor = 0.55
        shading.curvature_ridge_factor = 0.55
        shading.curvature_valley_factor = 0.55


def bbox_de_vertices(lista_V):
    """Calcula (mins, maxs) de una lista de listas de vértices."""
    mins = Vector((1e9, 1e9, 1e9))
    maxs = Vector((-1e9, -1e9, -1e9))
    for V in lista_V:
        for v in V:
            if v[0] < mins.x:
                mins.x = v[0]
            if v[1] < mins.y:
                mins.y = v[1]
            if v[2] < mins.z:
                mins.z = v[2]
            if v[0] > maxs.x:
                maxs.x = v[0]
            if v[1] > maxs.y:
                maxs.y = v[1]
            if v[2] > maxs.z:
                maxs.z = v[2]
    return mins, maxs


def posicionar_camara(cam_obj, mins, maxs, dir_vec, margen=1.22, aspect_x_y=1.5):
    """
    Posiciona la cámara en la dirección dir_vec mirando al centro de (mins, maxs)
    con distancia exacta para encuadrar toda la caja (lente 50mm).
    Devuelve la distancia de cámara empleada.
    """
    center = (mins + maxs) * 0.5
    corners = [
        Vector((x, y, z))
        for x in (mins.x, maxs.x)
        for y in (mins.y, maxs.y)
        for z in (mins.z, maxs.z)
    ]
    up_world = Vector((0.0, 0.0, 1.0))
    if abs(dir_vec.dot(up_world)) > 0.96:
        up_world = Vector((0.0, 1.0, 0.0))
    r = dir_vec.cross(up_world)
    if r.length < 1e-6:
        r = Vector((1.0, 0.0, 0.0))
    else:
        r.normalize()
    u = r.cross(dir_vec).normalized()

    # En Blender con sensor_fit = AUTO y sensor_width = 36mm, lente = 50mm:
    tan_x = 18.0 / 50.0
    tan_y = tan_x / max(aspect_x_y, 0.1)

    dist = 10.0
    for c in corners:
        dp = c - center
        x_cam = abs(dp.dot(r))
        y_cam = abs(dp.dot(u))
        z_cam = dp.dot(dir_vec)
        d_req = z_cam + margen * max(x_cam / tan_x, y_cam / tan_y)
        if d_req > dist:
            dist = d_req

    cam_obj.location = center + dir_vec * dist
    look_dir = center - cam_obj.location
    cam_obj.rotation_euler = look_dir.to_track_quat("-Z", "Y").to_euler()
    return dist


def activar_modo_objeto(obj, modo, grosor_alambre, mat_gris, mat_negro, mat_sombreado):
    """
    Configura los modificadores y materiales de obj para 'alambre' o 'sombreado'.
    """
    me = obj.data
    me.materials.clear()
    mod_wire = obj.modifiers.get("Wire")
    mod_subd = obj.modifiers.get("Subd")
    mod_split = obj.modifiers.get("EdgeSplit")

    if modo == "alambre":
        me.materials.append(mat_gris)
        me.materials.append(mat_negro)
        for p in me.polygons:
            p.use_smooth = False
        if mod_wire:
            mod_wire.show_render = True
            mod_wire.show_viewport = True
            mod_wire.thickness = grosor_alambre
        if mod_subd:
            mod_subd.show_render = False
            mod_subd.show_viewport = False
        if mod_split:
            mod_split.show_render = False
            mod_split.show_viewport = False
    else:
        me.materials.append(mat_sombreado)
        for p in me.polygons:
            p.use_smooth = True
        if mod_wire:
            mod_wire.show_render = False
            mod_wire.show_viewport = False
        if mod_subd:
            mod_subd.show_render = True
            mod_subd.show_viewport = True
        if mod_split:
            mod_split.show_render = True
            mod_split.show_viewport = True


def componer_imagen_cerca_lado_a_lado(ruta_alambre, ruta_sombreado, ruta_salida, titulo, subtitulo):
    """
    Combina lado a lado la imagen de cerca en alambre (izquierda) y sombreado con
    luz rasante (derecha) de un panel, agregando encabezado informativo con Pillow.
    """
    if not (os.path.exists(ruta_alambre) and os.path.exists(ruta_sombreado)):
        return False

    codigo_pil = r"""
import json, sys
from PIL import Image, ImageDraw, ImageFont

cfg = json.loads(sys.argv[1])
im_a = Image.open(cfg["alambre"]).convert("RGB")
im_s = Image.open(cfg["sombreado"]).convert("RGB")
w, h = im_a.size
pad = 14
header_h = 82
label_h = 34

total_w = 2 * w + 3 * pad
total_h = header_h + label_h + h + 2 * pad

canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

def cargar_fuente(tam, negrita=False):
    rutas = (
        ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"]
        if negrita
        else ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]
    )
    for r in rutas:
        try:
            return ImageFont.truetype(r, tam)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=tam)
    except Exception:
        return ImageFont.load_default()

f_title = cargar_fuente(22, negrita=True)
f_sub = cargar_fuente(15, negrita=False)
f_lbl = cargar_fuente(17, negrita=True)

draw.text((pad + 6, 14), cfg["titulo"], fill=(18, 18, 18), font=f_title)
draw.text((pad + 6, 46), cfg["subtitulo"], fill=(75, 75, 75), font=f_sub)

x_a = pad
x_s = 2 * pad + w
y_lbl = header_h
y_img = header_h + label_h

draw.text((x_a + 8, y_lbl + 6), "Alambre de cerca (jaula de control)", fill=(35, 35, 35), font=f_lbl)
draw.text((x_s + 8, y_lbl + 6), "Sombreado SubD de cerca (luz rasante para ver ondas)", fill=(35, 35, 35), font=f_lbl)

canvas.paste(im_a, (x_a, y_img))
canvas.paste(im_s, (x_s, y_img))

draw.rectangle([x_a, y_img, x_a + w - 1, y_img + h - 1], outline=(215, 215, 215), width=1)
draw.rectangle([x_s, y_img, x_s + w - 1, y_img + h - 1], outline=(215, 215, 215), width=1)

canvas.save(cfg["out_path"])
"""
    payload = json.dumps({
        "alambre": os.path.abspath(ruta_alambre),
        "sombreado": os.path.abspath(ruta_sombreado),
        "out_path": os.path.abspath(ruta_salida),
        "titulo": titulo,
        "subtitulo": subtitulo,
    })
    try:
        res = subprocess.run(
            ["python", "-I", "-c", codigo_pil, payload],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if res.returncode == 0 and os.path.exists(ruta_salida):
            return True
    except Exception:
        pass

    # Respaldo directo en Blender + numpy
    im_a = bpy.data.images.load(ruta_alambre)
    w, h = im_a.size
    px_a = np.array(im_a.pixels[:], dtype=np.float32).reshape((h, w, 4))
    bpy.data.images.remove(im_a)

    im_s = bpy.data.images.load(ruta_sombreado)
    px_s = np.array(im_s.pixels[:], dtype=np.float32).reshape((h, w, 4))
    bpy.data.images.remove(im_s)

    grid = np.concatenate([px_a, px_s], axis=1)
    H_tot, W_tot, _ = grid.shape
    out_im = bpy.data.images.new("CercaComp", width=W_tot, height=H_tot)
    out_im.pixels = grid.ravel().tolist()
    out_im.filepath_raw = os.path.abspath(ruta_salida)
    out_im.file_format = "PNG"
    out_im.save()
    bpy.data.images.remove(out_im)
    return os.path.exists(ruta_salida)


def evaluar_carpeta(carpeta_jaulas, datos_ref, salida_carpeta, crease_auto_deg, modo_v4=False):
    """
    Evalúa todos los pNN.obj en carpeta_jaulas contra datos_ref y genera las
    imágenes por pieza (incluyendo vistas de cerca con luz rasante) y del frente
    completo en salida_carpeta.
    """
    os.makedirs(salida_carpeta, exist_ok=True)
    scene, cam_obj, mat_gris, mat_negro, mat_subd, mat_rasante = configurar_escena_render()
    nombre_metodo = os.path.basename(os.path.normpath(carpeta_jaulas))

    archivos_obj = sorted(glob.glob(os.path.join(carpeta_jaulas, "p*.obj")))
    if not archivos_obj:
        archivos_obj = sorted(glob.glob(os.path.join(carpeta_jaulas, "*.obj")))

    objetos_escena = {}
    resultados_piezas = {}

    for ruta_obj in archivos_obj:
        nombre_archivo = os.path.basename(ruta_obj)
        pieza_id = os.path.splitext(nombre_archivo)[0]
        if pieza_id not in datos_ref:
            continue

        V_ref, F_ref, bvh_ref, bbox_pieza_ref, info_hd = datos_ref[pieza_id]
        V_raw, F_raw = leer_obj(ruta_obj)

        ruta_crease = os.path.join(carpeta_jaulas, f"{pieza_id}_crease.json")
        usar_json = os.path.exists(ruta_crease)
        crease_pairs = []
        if usar_json:
            try:
                with open(ruta_crease, "r", encoding="utf-8") as fh:
                    data_cj = json.load(fh)
                    crease_pairs = data_cj.get("crease", [])
            except Exception:
                usar_json = False

        bm, n_aristas_vivas = construir_bmesh_jaula(
            V_raw, F_raw, crease_pairs, usar_json, crease_auto_deg
        )
        metricas_malla = calcular_metricas_jaula(bm, len(F_raw))

        me = bpy.data.meshes.new(f"Mesh_{pieza_id}")
        bm.to_mesh(me)
        bm.free()
        me.update()

        obj = bpy.data.objects.new(pieza_id, me)
        bpy.context.collection.objects.link(obj)

        # Modificador Subdivision Surface (Catmull-Clark, nivel 3)
        mod_subd = obj.modifiers.new("Subd", "SUBSURF")
        mod_subd.subdivision_type = "CATMULL_CLARK"
        mod_subd.levels = 3
        mod_subd.render_levels = 3
        mod_subd.boundary_smooth = "PRESERVE_CORNERS"

        # Evaluar criterios 5, 8 y 9 sobre la malla subdividida nivel 3
        if modo_v4:
            ref4 = info_hd.get("ref_v4")
            if ref4 is None:
                print(f"[v4] {pieza_id}: sin referencia v4, se omite")
                bpy.data.objects.remove(obj, do_unlink=True)
                continue
            dist_s2r, dist_r2s, n_caras_subd, metricas_c8, metricas_c9 = evaluar_subd_v4(
                obj, V_ref, bvh_ref, ref4
            )
        else:
            dist_s2r, dist_r2s, n_caras_subd, metricas_c8, metricas_c9 = evaluar_subd_completo(
                obj, V_ref, bvh_ref, info_hd
            )

        # Modificador EdgeSplit para que las aristas vivas luzcan nítidas en el sombreado
        mod_split = obj.modifiers.new("EdgeSplit", "EDGE_SPLIT")
        mod_split.use_edge_angle = True
        mod_split.split_angle = math.radians(max(25.0, min(crease_auto_deg - 2.0, 35.0)))

        # Modificador Wireframe para el render de alambre de la jaula
        mod_wire = obj.modifiers.new("Wire", "WIREFRAME")
        mod_wire.use_replace = False
        mod_wire.use_even_offset = False
        mod_wire.use_relative_offset = False
        mod_wire.use_boundary = True
        mod_wire.material_offset = 1
        mod_wire.thickness = 0.35

        objetos_escena[pieza_id] = obj

        desc, tipo = NOMBRES_PIEZAS.get(pieza_id, (pieza_id, "panel"))
        c_sin_tri = (metricas_malla["triangulos"] == 0)
        c_ligera = (10 <= metricas_malla["caras"] <= 150)
        if modo_v4:
            # Contrato v4.1: el veredicto de C3 usa la proporción BRUTA (la efectiva se reporta aparte)
            c_sin_estiradas = (metricas_malla["proporcion_bruta_mayor_3_5_pct"] <= 10.0)
        else:
            c_sin_estiradas = (metricas_malla["proporcion_mayor_3_5_pct"] <= 10.0)
        c_sin_polos = (metricas_malla["polos_interiores_6_mas"] == 0)
        c_fiel = (
            dist_s2r["media"] <= 0.3
            and dist_s2r["p95"] <= 1.5
            and dist_r2s["p95"] <= 1.5
        )
        c_1_a_5 = bool(c_sin_tri and c_ligera and c_sin_estiradas and c_sin_polos and c_fiel)
        if modo_v4:
            # N/D (referencia sin líneas o sin regiones) no aprueba ni reprueba: no entra al "cumple todo"
            c_aristas_diseno = metricas_c8["cumple"]
            c_planitud = metricas_c9["cumple"]
        else:
            c_aristas_diseno = bool(metricas_c8["cumple"])
            c_planitud = bool(metricas_c9["cumple"])

        resultados_piezas[pieza_id] = {
            "archivo": nombre_archivo,
            "nombre": desc,
            "tipo": tipo,
            **metricas_malla,
            "aristas_vivas": n_aristas_vivas,
            "fuente_crease": "json" if usar_json else f"auto_{int(crease_auto_deg)}",
            "caras_subd_niv3": n_caras_subd,
            "dist_subd_a_ref": dist_s2r,
            "dist_ref_a_subd": dist_r2s,
            "aristas_diseno": metricas_c8,
            "planitud_regiones": metricas_c9,
            "criterios": {
                "cuadros_sin_triangulos": c_sin_tri,
                "ligera_10_150": c_ligera,
                "sin_estiradas_90pct_le_3_5": c_sin_estiradas,
                "sin_polos_6_mas": c_sin_polos,
                "fiel_hd": c_fiel,
                "cumple_1_a_5": c_1_a_5,
                "aristas_diseno_respetadas": c_aristas_diseno,
                "caras_planas_sin_ondas": c_planitud,
                "cumple_todo": bool(c_1_a_5 and c_aristas_diseno is not False and c_planitud is not False),
            },
        }
        if modo_v4:
            ref4 = info_hd["ref_v4"]
            resultados_piezas[pieza_id]["contrato"] = "referencia_v4.1"
            resultados_piezas[pieza_id]["referencia_v4"] = {
                "archivo": ref4["ruta"], "sha256_hd": ref4["sha256_hd"], "sha_coincide": ref4["sha_coincide"],
            }
            resultados_piezas[pieza_id]["c3_bruta_pct"] = metricas_malla["proporcion_bruta_mayor_3_5_pct"]
            resultados_piezas[pieza_id]["c3_efectiva_pct"] = metricas_malla["proporcion_mayor_3_5_pct"]
            resultados_piezas[pieza_id]["c5_media_inversa_mm"] = dist_r2s["media"]

    # 1. Renderizar imágenes por pieza (3 cámaras estándar 1200x800 + vista de cerca con luz rasante)
    for pieza_id, obj in objetos_escena.items():
        for otro_id, otro_obj in objetos_escena.items():
            otro_obj.hide_render = (otro_id != pieza_id)

        mins_p, maxs_p = datos_ref[pieza_id][3]
        info_hd = datos_ref[pieza_id][4]

        # 3 cámaras estándar (1200x800)
        scene.render.resolution_x = 1200
        scene.render.resolution_y = 800
        for modo in ("alambre", "sombreado"):
            activar_iluminacion_escena(scene, rasante=False, modo=modo)
            for cam_id, _, dir_vec in CAMARAS:
                dist = posicionar_camara(cam_obj, mins_p, maxs_p, dir_vec, margen=1.22, aspect_x_y=1.5)
                grosor = max(0.04, 0.00085 * dist)
                activar_modo_objeto(obj, modo, grosor, mat_gris, mat_negro, mat_subd)
                ruta_img = os.path.abspath(os.path.join(salida_carpeta, f"{pieza_id}_{modo}_{cam_id}.png"))
                scene.render.filepath = ruta_img
                bpy.ops.render.render(write_still=True)

        # Vista de cerca por panel (alambre y sombreado con luz rasante lado a lado)
        # Orientamos la cámara mirando casi de frente a la superficie principal del panel
        n_dom = info_hd["n_dom"]
        dir_oblicua = Vector((0.40, -0.55, 0.30)).normalized()
        dir_cerca = (n_dom * 0.78 + dir_oblicua * 0.22).normalized()

        scene.render.resolution_x = 960
        scene.render.resolution_y = 800
        dist_cerca = posicionar_camara(cam_obj, mins_p, maxs_p, dir_cerca, margen=1.06, aspect_x_y=1.2)
        grosor_cerca = max(0.035, 0.00082 * dist_cerca)

        ruta_alambre_cerca = os.path.abspath(os.path.join(salida_carpeta, f"{pieza_id}_alambre_cerca.png"))
        activar_iluminacion_escena(scene, rasante=False, modo="alambre")
        activar_modo_objeto(obj, "alambre", grosor_cerca, mat_gris, mat_negro, mat_subd)
        scene.render.filepath = ruta_alambre_cerca
        bpy.ops.render.render(write_still=True)

        ruta_sombreado_cerca = os.path.abspath(os.path.join(salida_carpeta, f"{pieza_id}_sombreado_cerca.png"))
        activar_iluminacion_escena(scene, rasante=True, modo="sombreado")
        activar_modo_objeto(obj, "sombreado", grosor_cerca, mat_gris, mat_negro, mat_rasante)
        scene.render.filepath = ruta_sombreado_cerca
        bpy.ops.render.render(write_still=True)

        p_res = resultados_piezas[pieza_id]
        c8 = p_res["aristas_diseno"]
        c9 = p_res["planitud_regiones"]
        titulo_cerca = f"{pieza_id} ({p_res['nombre']}) — Método: {nombre_metodo}"
        subtitulo_cerca = (
            f"Caras: {p_res['caras']} ({p_res['cuadros_pct']:.0f}% cuad) | "
            f"SubD->HD p95: {p_res['dist_subd_a_ref']['p95']:.2f} mm | "
            f"Crit.8 Aristas p95: {_fmt(c8['dist_p95_mm'])} mm (cob {_fmt(c8['cobertura_05mm_pct'], 0)}%) | "
            f"Crit.9 Planitud p95 máx: {_fmt(c9['p95_max_mm'])} mm (med {_fmt(c9['p95_media_mm'])} mm)"
        )
        ruta_cerca_lado = os.path.abspath(os.path.join(salida_carpeta, f"{pieza_id}_cerca.png"))
        componer_imagen_cerca_lado_a_lado(
            ruta_alambre_cerca, ruta_sombreado_cerca, ruta_cerca_lado, titulo_cerca, subtitulo_cerca
        )

    # 2. Renderizar imágenes del frente completo (todas las piezas juntas, 1200x800)
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 800
    if objetos_escena:
        for obj in objetos_escena.values():
            obj.hide_render = False

        mins_glob, maxs_glob = datos_ref["_global_bbox"]
        for modo in ("alambre", "sombreado"):
            activar_iluminacion_escena(scene, rasante=False, modo=modo)
            for cam_id, _, dir_vec in CAMARAS:
                dist = posicionar_camara(cam_obj, mins_glob, maxs_glob, dir_vec, margen=1.18, aspect_x_y=1.5)
                grosor = max(0.08, 0.00080 * dist)
                for obj in objetos_escena.values():
                    activar_modo_objeto(obj, modo, grosor, mat_gris, mat_negro, mat_subd)
                ruta_img = os.path.abspath(os.path.join(salida_carpeta, f"frente_{modo}_{cam_id}.png"))
                scene.render.filepath = ruta_img
                bpy.ops.render.render(write_still=True)

    resumen_todas = sintetizar_grupo(list(resultados_piezas.values()))
    paneles_lista = [v for v in resultados_piezas.values() if v.get("tipo") == "panel"]
    resumen_paneles = sintetizar_grupo(paneles_lista if paneles_lista else list(resultados_piezas.values()))

    return {
        "carpeta_jaulas": carpeta_jaulas,
        "metodo": nombre_metodo,
        "crease_auto_grados": crease_auto_deg,
        "resumen": resumen_todas,
        "resumen_paneles": resumen_paneles,
        "piezas": resultados_piezas,
    }


def _fmt(v, dec=2):
    """Número con decimales o 'N/D' si no aplica (contrato v4.1)."""
    if v is None:
        return "N/D"
    return f"{v:.{dec}f}"


def _media_nd(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _max_nd(vals):
    vals = [v for v in vals if v is not None]
    return float(np.max(vals)) if vals else None


def _r(v, d=4):
    return round(v, d) if v is not None else None


def sintetizar_grupo(lista_piezas):
    if not lista_piezas:
        return {}
    n = len(lista_piezas)
    return {
        "num_piezas": n,
        "caras_total": int(sum(p["caras"] for p in lista_piezas)),
        "caras_media": round(float(np.mean([p["caras"] for p in lista_piezas])), 1),
        "caras_min": int(min(p["caras"] for p in lista_piezas)),
        "caras_max": int(max(p["caras"] for p in lista_piezas)),
        "cuadros_pct_media": round(float(np.mean([p["cuadros_pct"] for p in lista_piezas])), 2),
        "triangulos_pct_media": round(float(np.mean([p["triangulos_pct"] for p in lista_piezas])), 2),
        "ngonos_pct_media": round(float(np.mean([p["ngonos_pct"] for p in lista_piezas])), 2),
        "proporcion_mayor_3_5_pct_media": round(
            float(np.mean([p["proporcion_mayor_3_5_pct"] for p in lista_piezas])), 2
        ),
        "polos_interiores_6_mas_total": int(sum(p["polos_interiores_6_mas"] for p in lista_piezas)),
        "polos_interiores_6_mas_media": round(
            float(np.mean([p["polos_interiores_6_mas"] for p in lista_piezas])), 2
        ),
        "dist_subd_a_ref_media": round(
            float(np.mean([p["dist_subd_a_ref"]["media"] for p in lista_piezas])), 4
        ),
        "dist_subd_a_ref_p95_media": round(
            float(np.mean([p["dist_subd_a_ref"]["p95"] for p in lista_piezas])), 4
        ),
        "dist_subd_a_ref_p95_max": round(
            float(np.max([p["dist_subd_a_ref"]["p95"] for p in lista_piezas])), 4
        ),
        "dist_subd_a_ref_max": round(
            float(np.max([p["dist_subd_a_ref"]["max"] for p in lista_piezas])), 4
        ),
        "dist_ref_a_subd_media": round(
            float(np.mean([p["dist_ref_a_subd"]["media"] for p in lista_piezas])), 4
        ),
        "dist_ref_a_subd_p95_media": round(
            float(np.mean([p["dist_ref_a_subd"]["p95"] for p in lista_piezas])), 4
        ),
        "dist_ref_a_subd_p95_max": round(
            float(np.max([p["dist_ref_a_subd"]["p95"] for p in lista_piezas])), 4
        ),
        "dist_ref_a_subd_max": round(
            float(np.max([p["dist_ref_a_subd"]["max"] for p in lista_piezas])), 4
        ),
        "aristas_diseno_p95_media": _r(_media_nd([p["aristas_diseno"]["dist_p95_mm"] for p in lista_piezas])),
        "aristas_diseno_p95_max": _r(_max_nd([p["aristas_diseno"]["dist_p95_mm"] for p in lista_piezas])),
        "aristas_diseno_cobertura_media_pct": _r(_media_nd([p["aristas_diseno"]["cobertura_05mm_pct"] for p in lista_piezas]), 2),
        "aristas_diseno_cumplen": int(
            sum(1 for p in lista_piezas if p["criterios"]["aristas_diseno_respetadas"] is True)
        ),
        "aristas_diseno_nd": int(
            sum(1 for p in lista_piezas if p["criterios"]["aristas_diseno_respetadas"] is None)
        ),
        "planitud_p95_media": _r(_media_nd([p["planitud_regiones"]["p95_media_mm"] for p in lista_piezas])),
        "planitud_p95_max_media": _r(_media_nd([p["planitud_regiones"]["p95_max_mm"] for p in lista_piezas])),
        "planitud_p95_max": _r(_max_nd([p["planitud_regiones"]["p95_max_mm"] for p in lista_piezas])),
        "planitud_cumplen": int(
            sum(1 for p in lista_piezas if p["criterios"]["caras_planas_sin_ondas"] is True)
        ),
        "piezas_cumplen_1_a_5": int(
            sum(1 for p in lista_piezas if p["criterios"]["cumple_1_a_5"])
        ),
        "piezas_cumplen_todo": int(
            sum(1 for p in lista_piezas if p["criterios"]["cumple_todo"])
        ),
    }


def cargar_referencias_hd(carpeta_ref, carpeta_lineas="salidas/lineas", solo=None, carpeta_v4=None):
    """
    Carga todas las mallas de referencia HD, construye sus BVHTree y bounding boxes,
    y extrae sus líneas de doblez y regiones planas (para los criterios 8 y 9).
    """
    datos_ref = {}
    todas_V = []
    rutas_ref = sorted(glob.glob(os.path.join(carpeta_ref, "p*.obj"))) or sorted(glob.glob(os.path.join(carpeta_ref, "*.obj")))  # v5: piezas sNNN
    for ruta in rutas_ref:
        pieza_id = os.path.splitext(os.path.basename(ruta))[0]
        if solo and pieza_id not in solo:
            continue
        V, F = leer_obj(ruta)
        bvh = BVHTree.FromPolygons(V, F)
        mins, maxs = bbox_de_vertices([V])
        if carpeta_v4:
            # Contrato v4.1: las líneas y regiones vienen SOLO de la referencia congelada
            info_hd = info_hd_minima(V, F)
            info_hd["ref_v4"] = cargar_referencia_v4(carpeta_v4, pieza_id, ruta, V)
            r4 = info_hd["ref_v4"]
            if r4 is None:
                print(f"[v4] {pieza_id}: no hay {pieza_id}.json en {carpeta_v4}")
            elif not r4["sha_coincide"]:
                print(f"[v4] AVISO {pieza_id}: el sha256 de la HD no coincide con la referencia")
        else:
            ruta_json_lin = os.path.join(carpeta_lineas, f"{pieza_id}.json") if carpeta_lineas else None
            info_hd = extraer_lineas_y_regiones_hd(V, F, ruta_json_lin)
        datos_ref[pieza_id] = (V, F, bvh, (mins, maxs), info_hd)
        todas_V.append(V)
    if todas_V:
        datos_ref["_global_bbox"] = bbox_de_vertices(todas_V)
    else:
        datos_ref["_global_bbox"] = (Vector((-100, -100, -100)), Vector((100, 100, 100)))
    return datos_ref


def generar_hoja_comparacion(columnas_info, ruta_salida, cam_id="34"):
    """
    Genera una hoja de comparación donde:
      - Columnas = métodos (datos/base_tripo, datos/base_qremeshify y cualquier carpeta pasada)
      - Filas = [alambre, sombreado]
    """
    items = []
    for col in columnas_info:
        img_alambre = os.path.join(col["carpeta_imgs"], f"frente_alambre_{cam_id}.png")
        img_sombreado = os.path.join(col["carpeta_imgs"], f"frente_sombreado_{cam_id}.png")
        if os.path.exists(img_alambre) and os.path.exists(img_sombreado):
            items.append({
                "titulo": col["titulo"],
                "subtitulo": col["subtitulo"],
                "alambre": img_alambre,
                "sombreado": img_sombreado,
            })
    if not items:
        return False

    codigo_pil = r"""
import json, sys
from PIL import Image, ImageDraw, ImageFont

cfg = json.loads(sys.argv[1])
items = cfg["items"]
out_path = cfg["out_path"]

imgs_a = [Image.open(it["alambre"]).convert("RGB") for it in items]
imgs_s = [Image.open(it["sombreado"]).convert("RGB") for it in items]

w, h = imgs_a[0].size
n_cols = len(items)
header_h = 90
row_label_w = 56
pad = 16

total_w = row_label_w + n_cols * w + (n_cols + 1) * pad
total_h = header_h + 2 * h + 3 * pad

canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

def cargar_fuente(tam, negrita=False):
    rutas = (
        ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"]
        if negrita
        else ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]
    )
    for r in rutas:
        try:
            return ImageFont.truetype(r, tam)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=tam)
    except Exception:
        return ImageFont.load_default()

f_title = cargar_fuente(24, negrita=True)
f_sub = cargar_fuente(15, negrita=False)
f_row = cargar_fuente(20, negrita=True)

for idx, it in enumerate(items):
    x0 = row_label_w + pad + idx * (w + pad)
    y_a = header_h + pad
    y_s = header_h + 2 * pad + h

    draw.text((x0 + 12, 18), it["titulo"], fill=(20, 20, 20), font=f_title)
    draw.text((x0 + 12, 52), it["subtitulo"], fill=(80, 80, 80), font=f_sub)

    canvas.paste(imgs_a[idx], (x0, y_a))
    canvas.paste(imgs_s[idx], (x0, y_s))

    draw.rectangle([x0, y_a, x0 + w - 1, y_a + h - 1], outline=(225, 225, 225), width=1)
    draw.rectangle([x0, y_s, x0 + w - 1, y_s + h - 1], outline=(225, 225, 225), width=1)

for texto, y_top in [("Alambre (jaula)", header_h + pad), ("Sombreado (SubD)", header_h + 2 * pad + h)]:
    txt_img = Image.new("RGB", (h, row_label_w), (255, 255, 255))
    d_txt = ImageDraw.Draw(txt_img)
    d_txt.text((h // 2 - 80, 16), texto, fill=(40, 40, 40), font=f_row)
    txt_rot = txt_img.rotate(90, expand=True)
    canvas.paste(txt_rot, (4, y_top))

canvas.save(out_path)
"""
    ruta_salida = os.path.abspath(ruta_salida)
    payload = json.dumps({"items": items, "out_path": ruta_salida})
    try:
        res = subprocess.run(
            ["python", "-I", "-c", codigo_pil, payload],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if res.returncode == 0 and os.path.exists(ruta_salida):
            return True
    except Exception:
        pass

    arrays_a = []
    arrays_s = []
    for it in items:
        im_a = bpy.data.images.load(it["alambre"])
        w, h = im_a.size
        px_a = np.array(im_a.pixels[:], dtype=np.float32).reshape((h, w, 4))
        bpy.data.images.remove(im_a)

        im_s = bpy.data.images.load(it["sombreado"])
        px_s = np.array(im_s.pixels[:], dtype=np.float32).reshape((h, w, 4))
        bpy.data.images.remove(im_s)

        arrays_a.append(px_a)
        arrays_s.append(px_s)

    fila_a = np.concatenate(arrays_a, axis=1)
    fila_s = np.concatenate(arrays_s, axis=1)
    grid = np.concatenate([fila_s, fila_a], axis=0)
    H_tot, W_tot, _ = grid.shape
    out_im = bpy.data.images.new("HojaComp", width=W_tot, height=H_tot)
    out_im.pixels = grid.ravel().tolist()
    out_im.filepath_raw = ruta_salida
    out_im.file_format = "PNG"
    out_im.save()
    bpy.data.images.remove(out_im)
    return os.path.exists(ruta_salida)


def generar_hoja_cerca_paneles(columnas_info, paneles_ids, ruta_salida):
    """
    Genera una hoja comparativa de cerca con luz rasante donde cada fila es un panel
    y cada columna muestra el par (alambre + sombreado rasante) de un método.
    """
    filas = []
    for pid in paneles_ids:
        celdas = []
        for col in columnas_info:
            img_cerca = os.path.join(col["carpeta_imgs"], f"{pid}_cerca.png")
            if os.path.exists(img_cerca):
                celdas.append(os.path.abspath(img_cerca))
        if len(celdas) == len(columnas_info) and celdas:
            desc, _ = NOMBRES_PIEZAS.get(pid, (pid, "panel"))
            filas.append({"pid": pid, "desc": desc, "celdas": celdas})

    if not filas:
        return False

    codigo_pil = r"""
import json, sys
from PIL import Image, ImageDraw

cfg = json.loads(sys.argv[1])
filas = cfg["filas"]
out_path = cfg["out_path"]

sample = Image.open(filas[0]["celdas"][0]).convert("RGB")
cw, ch = sample.size
scale = 0.5
tw, th = int(cw * scale), int(ch * scale)
n_cols = len(filas[0]["celdas"])
n_rows = len(filas)
pad = 12

canvas = Image.new("RGB", (n_cols * tw + (n_cols + 1) * pad, n_rows * th + (n_rows + 1) * pad), (248, 248, 250))
draw = ImageDraw.Draw(canvas)

for r_idx, fila in enumerate(filas):
    y0 = pad + r_idx * (th + pad)
    for c_idx, ruta in enumerate(fila["celdas"]):
        x0 = pad + c_idx * (tw + pad)
        im = Image.open(ruta).convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
        canvas.paste(im, (x0, y0))
        draw.rectangle([x0, y0, x0 + tw - 1, y0 + th - 1], outline=(210, 210, 215), width=1)

canvas.save(out_path)
"""
    payload = json.dumps({"filas": filas, "out_path": os.path.abspath(ruta_salida)})
    try:
        res = subprocess.run(
            ["python", "-I", "-c", codigo_pil, payload],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if res.returncode == 0 and os.path.exists(ruta_salida):
            return True
    except Exception:
        pass
    return False


def formatear_tabla_metodo(eval_data):
    """Genera la tabla Markdown detallada por pieza para un método (incluyendo criterios 8 y 9)."""
    lineas = []
    metodo = eval_data["metodo"]
    carpeta = eval_data["carpeta_jaulas"]
    r_pan = eval_data["resumen_paneles"]
    r_tot = eval_data["resumen"]

    lineas.append(f"## Método: `{metodo}` (`{carpeta}`)")
    lineas.append("")
    lineas.append(
        "| Pieza | Descripción | Tipo | Caras | % Cuadros | % Triáng. | % Prop > 3.5 | Polos ≥ 6 | Creases | "
        "SubD→HD media (mm) | SubD→HD p95 (mm) | HD→SubD p95 (mm) | "
        "Crit. 8: Aristas p95 (mm) | Crit. 8: Cob. ≤0.5mm | "
        "Crit. 9: Planitud p95 med / máx (mm) | Crit. 9: Reg. planas | Cumple 1–5 | Cumple 1–9 |"
    )
    lineas.append(
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|"
    )

    for pid in sorted(eval_data["piezas"].keys()):
        p = eval_data["piezas"][pid]
        s2r = p["dist_subd_a_ref"]
        r2s = p["dist_ref_a_subd"]
        c8 = p["aristas_diseno"]
        c9 = p["planitud_regiones"]
        ok_15 = "Sí" if p["criterios"]["cumple_1_a_5"] else "No"
        ok_all = "Sí" if p["criterios"]["cumple_todo"] else "No"
        lineas.append(
            f"| `{pid}` | {p['nombre']} | {p['tipo']} | {p['caras']} | {p['cuadros_pct']:.1f}% | "
            f"{p['triangulos_pct']:.1f}% | {p['proporcion_mayor_3_5_pct']:.1f}% | "
            f"{p['polos_interiores_6_mas']} | {p['aristas_vivas']} ({p['fuente_crease']}) | "
            f"{s2r['media']:.3f} | {s2r['p95']:.3f} | {r2s['p95']:.3f} | "
            f"{c8['dist_p95_mm']:.3f} | {c8['cobertura_05mm_pct']:.1f}% | "
            f"{c9['p95_media_mm']:.3f} / {c9['p95_max_mm']:.3f} | "
            f"{c9['regiones_cumplen']}/{c9['num_regiones_planas']} | {ok_15} | {ok_all} |"
        )

    if r_pan:
        lineas.append(
            f"| **Media (10 paneles)** | — | panel | **{r_pan['caras_media']:.1f}** (tot {r_pan['caras_total']}) | "
            f"**{r_pan['cuadros_pct_media']:.1f}%** | **{r_pan['triangulos_pct_media']:.1f}%** | "
            f"**{r_pan['proporcion_mayor_3_5_pct_media']:.1f}%** | **{r_pan['polos_interiores_6_mas_total']}** tot | — | "
            f"**{r_pan['dist_subd_a_ref_media']:.3f}** | **{r_pan['dist_subd_a_ref_p95_media']:.3f}** (máx {r_pan['dist_subd_a_ref_p95_max']:.3f}) | "
            f"**{r_pan['dist_ref_a_subd_p95_media']:.3f}** (máx {r_pan['dist_ref_a_subd_p95_max']:.3f}) | "
            f"**{r_pan['aristas_diseno_p95_media']:.3f}** (máx {r_pan['aristas_diseno_p95_max']:.3f}) | "
            f"**{r_pan['aristas_diseno_cobertura_media_pct']:.1f}%** ({r_pan['aristas_diseno_cumplen']}/{r_pan['num_piezas']}) | "
            f"**{r_pan['planitud_p95_media']:.3f} / {r_pan['planitud_p95_max_media']:.3f}** (máx {r_pan['planitud_p95_max']:.3f}) | "
            f"**{r_pan['planitud_cumplen']}/{r_pan['num_piezas']}** | "
            f"**{r_pan['piezas_cumplen_1_a_5']}/{r_pan['num_piezas']}** | "
            f"**{r_pan['piezas_cumplen_todo']}/{r_pan['num_piezas']}** |"
        )
    if r_tot and r_tot.get("num_piezas", 0) != r_pan.get("num_piezas", 0):
        lineas.append(
            f"| **Media (todas: {r_tot['num_piezas']})** | — | todas | **{r_tot['caras_media']:.1f}** (tot {r_tot['caras_total']}) | "
            f"**{r_tot['cuadros_pct_media']:.1f}%** | **{r_tot['triangulos_pct_media']:.1f}%** | "
            f"**{r_tot['proporcion_mayor_3_5_pct_media']:.1f}%** | **{r_tot['polos_interiores_6_mas_total']}** tot | — | "
            f"**{r_tot['dist_subd_a_ref_media']:.3f}** | **{r_tot['dist_subd_a_ref_p95_media']:.3f}** | "
            f"**{r_tot['dist_ref_a_subd_p95_media']:.3f}** | "
            f"**{r_tot['aristas_diseno_p95_media']:.3f}** | **{r_tot['aristas_diseno_cobertura_media_pct']:.1f}%** | "
            f"**{r_tot['planitud_p95_media']:.3f} / {r_tot['planitud_p95_max_media']:.3f}** | "
            f"**{r_tot['planitud_cumplen']}/{r_tot['num_piezas']}** | "
            f"**{r_tot['piezas_cumplen_1_a_5']}/{r_tot['num_piezas']}** | "
            f"**{r_tot['piezas_cumplen_todo']}/{r_tot['num_piezas']}** |"
        )
    lineas.append("")
    return "\n".join(lineas)


def escribir_resumen_md(evaluaciones, ruta_md):
    """Escribe el informe resumen.md con la comparativa global (criterios 1 a 9) y tablas por pieza."""
    lineas = [
        "# Evaluación de Jaulas SubD contra Referencia HD (`datos/hd`) — Vuelta 2",
        "",
        "Criterios de aceptación por panel (`ENCARGO.md`):",
        "1. **Cuadros**: 0 % de triángulos en la jaula.",
        "2. **Ligera**: de 10 a 150 caras por panel.",
        "3. **Sin caras estiradas**: proporción (arista mayor / arista menor) > 3.5 en ≤ 10 % de las caras.",
        "4. **Sin polos raros**: 0 vértices interiores con 6 o más aristas.",
        "5. **Fiel**: superficie subdividida (Catmull-Clark nivel 3) a ≤ 0.3 mm de media y ≤ 1.5 mm en p95 de `datos/hd`, y de vuelta (HD→SubD) p95 ≤ 1.5 mm.",
        "6–7. **Paneles planos y aristas vivas** donde el diseño dobla.",
        "8. **Aristas del diseño respetadas**: las líneas de doblez de la referencia HD (donde la superficie dobla más de 25–30°) caen sobre aristas vivas de la superficie subdividida: **p95 ≤ 0.5 mm** y **≥ 90 % de su largo cubierto** (a ≤ 0.5 mm).",
        "9. **Caras planas sin ondas (planitud)**: en cada región plana de la referencia, la superficie subdividida se desvía de su plano **≤ 0.2 mm en el p95**.",
        "",
        "## Comparación global (sobre los 10 paneles)",
        "",
        "| Método | Carpeta | Paneles | Caras/panel (media [mín–máx]) | % Cuadros | % Prop > 3.5 | Polos ≥ 6 | "
        "SubD→HD media / p95 med (mm) | HD→SubD p95 med (mm) | "
        "Crit. 8: Aristas p95 med (mm) | Crit. 8: Cob. media (%) | Crit. 8: Pasan | "
        "Crit. 9: Planitud p95 med / máx-med (mm) | Crit. 9: Pasan | Cumplen 1–5 | Cumplen 1–9 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|:---:|:---:|",
    ]

    for ev in evaluaciones:
        rp = ev["resumen_paneles"]
        if not rp:
            continue
        lineas.append(
            f"| **{ev['metodo']}** | `{ev['carpeta_jaulas']}` | {rp['num_piezas']} | "
            f"{rp['caras_media']:.1f} [{rp['caras_min']}–{rp['caras_max']}] | "
            f"{rp['cuadros_pct_media']:.1f}% | "
            f"{rp['proporcion_mayor_3_5_pct_media']:.1f}% | {rp['polos_interiores_6_mas_total']} | "
            f"{rp['dist_subd_a_ref_media']:.3f} / {rp['dist_subd_a_ref_p95_media']:.3f} | "
            f"{rp['dist_ref_a_subd_p95_media']:.3f} | "
            f"{rp['aristas_diseno_p95_media']:.3f} | "
            f"{rp['aristas_diseno_cobertura_media_pct']:.1f}% | "
            f"{rp['aristas_diseno_cumplen']}/{rp['num_piezas']} | "
            f"{rp['planitud_p95_media']:.3f} / {rp['planitud_p95_max_media']:.3f} | "
            f"{rp['planitud_cumplen']}/{rp['num_piezas']} | "
            f"{rp['piezas_cumplen_1_a_5']}/{rp['num_piezas']} | "
            f"**{rp['piezas_cumplen_todo']}/{rp['num_piezas']}** |"
        )

    lineas.append("")
    for ev in evaluaciones:
        lineas.append(formatear_tabla_metodo(ev))

    lineas.append("## Imágenes generadas y vistas de cerca con luz rasante")
    lineas.append("")
    lineas.append("- **Imágenes de cerca por panel con luz rasante**:")
    lineas.append("  - `<pieza>_cerca.png` (y `<metodo>/<pieza>_cerca.png`): alambre de la jaula y sombreado SubD con **luz rasante** lado a lado por panel, con métricas de los criterios 8 y 9 en el encabezado.")
    lineas.append("  - `<pieza>_alambre_cerca.png` y `<pieza>_sombreado_cerca.png`: vistas individuales de cerca por panel.")
    lineas.append("  - `hoja_cerca_paneles.png`: mosaico comparativo de todos los paneles de cerca con luz rasante entre todos los métodos evaluados.")
    lineas.append("- **Hojas de comparación del frente completo**:")
    lineas.append("  - `hoja_comparacion.png` (y `hoja_comparacion_34.png`, `comparacion.png`): vista 3/4 de frente (`34`), columnas = métodos, filas = alambre y sombreado SubD.")
    lineas.append("  - `hoja_comparacion_costado.png` y `hoja_comparacion_frente.png`: vistas lateral y frontal.")
    lineas.append("")

    with open(ruta_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas))


def _sino_nd(ok, e8, e9):
    if not ok:
        return "no"
    nd = [n for n, e in (("C8", e8), ("C9", e9)) if e == "N/D"]
    return f"sí ({', '.join(nd)} N/D)" if nd else "sí"


def escribir_resumen_v4_md(evaluaciones, ruta_md, args):
    """Resumen con el contrato referencia_v4.1. No se mezcla con los resúmenes históricos."""
    lineas = [
        "# Banco v4.1: jaulas contra la referencia congelada",
        "",
        f"- Referencia: `{args.referencia_v4}` (contrato `referencia_v4.1`, solo de la HD).",
        f"- Piezas: {', '.join(args.solo) if args.solo else 'todas las que tengan referencia'}.",
        "- C3: veredicto con la proporción **bruta** (arista mayor / menor > 3.5 en ≤ 10 % de caras); la efectiva va entre paréntesis.",
        "- C5: SubD→HD media ≤ 0.3 y p95 ≤ 1.5; HD→SubD p95 ≤ 1.5. La media inversa (HD→SubD) se reporta, no entra al veredicto.",
        "- C8: líneas de doblez de la referencia contra los crease **interiores** de la SubD nivel 3 (las aristas de borde no cuentan): p95 ≤ 0.5 mm y ≥ 90 % del largo a ≤ 0.5 mm. Sin líneas = N/D.",
        "- C9: por región plana, la SubD que cae sobre ella (proyección más cercana a la HD) contra el plano **fijo**: peor p95 por región ≤ 0.2 mm, sin recentrar ni promediar. Región sin puntos SubD = falla.",
        "- N/D no aprueba ni reprueba: queda fuera del «cumple 1–9».",
        "- Estas cifras no se comparan con los resúmenes de las vueltas 1 a 3 (otra referencia).",
        "",
    ]
    for ev in evaluaciones:
        lineas.append(f"## `{ev['metodo']}` (`{ev['carpeta_jaulas']}`)")
        lineas.append("")
        lineas.append(
            "| Pieza | Caras | % cuadros | C3 bruta (efectiva) | Polos ≥6 | C5 SubD→HD media / p95 | C5 HD→SubD media / p95 | "
            "C8 p95 / cobertura | C9 peor p95 (regiones que pasan) | C1–5 | C8 | C9 | C1–9 |"
        )
        lineas.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|")
        for pid in sorted(ev["piezas"].keys()):
            p = ev["piezas"][pid]
            c8, c9 = p["aristas_diseno"], p["planitud_regiones"]
            s2r, r2s = p["dist_subd_a_ref"], p["dist_ref_a_subd"]
            cr = p["criterios"]
            c8_txt = "N/D" if c8["estado"] == "N/D" else f"{_fmt(c8['dist_p95_mm'])} / {_fmt(c8['cobertura_05mm_pct'], 1)} %"
            c9_txt = "N/D" if c9["estado"] == "N/D" else (
                f"{_fmt(c9['p95_max_mm'])} ({c9['regiones_cumplen']}/{c9['num_regiones_planas']})")
            lineas.append(
                f"| `{pid}` | {p['caras']} | {p['cuadros_pct']:.1f} | {p['c3_bruta_pct']:.1f} % ({p['c3_efectiva_pct']:.1f} %) | "
                f"{p['polos_interiores_6_mas']} | {s2r['media']:.3f} / {s2r['p95']:.3f} | {r2s['media']:.3f} / {r2s['p95']:.3f} | "
                f"{c8_txt} | {c9_txt} | {'sí' if cr['cumple_1_a_5'] else 'no'} | {c8['estado']} | {c9['estado']} | "
                f"{_sino_nd(cr['cumple_todo'], c8['estado'], c9['estado'])} |"
            )
        lineas.append("")
        lineas.append("C9 por región (id: p95 mm / puntos SubD; residuo de la HD a su plano):")
        lineas.append("")
        for pid in sorted(ev["piezas"].keys()):
            c9 = ev["piezas"][pid]["planitud_regiones"]
            if c9["estado"] == "N/D":
                lineas.append(f"- `{pid}`: sin regiones planas en la referencia (N/D).")
                continue
            partes = [
                f"{r['id']}: {_fmt(r['p95_mm'])} / {r['n_puntos']} (HD {r['residuo_hd_p95_mm']})"
                for r in c9["por_region"]
            ]
            lineas.append(f"- `{pid}`: " + "; ".join(partes))
        lineas.append("")
    with open(ruta_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas))


def parsear_argumentos():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Evaluador de jaulas SubD en Blender 5.2 (Vuelta 2)")
    parser.add_argument(
        "--jaulas",
        nargs="+",
        action="append",
        required=True,
        help="Carpeta(s) con archivos pNN.obj de jaulas a evaluar",
    )
    parser.add_argument(
        "--ref",
        default="datos/hd",
        help="Carpeta con las mallas de referencia HD (por defecto: datos/hd)",
    )
    parser.add_argument(
        "--lineas",
        default="salidas/lineas",
        help="Carpeta opcional con los JSON de líneas y regiones de diseño (por defecto: salidas/lineas)",
    )
    parser.add_argument(
        "--salida",
        required=True,
        help="Carpeta donde se escribirán resultados.json, resumen.md e imágenes",
    )
    parser.add_argument(
        "--crease-auto",
        type=float,
        default=30.0,
        help="Umbral en grados para marcar aristas vivas si no existe pNN_crease.json (por defecto: 30)",
    )
    parser.add_argument(
        "--referencia-v4",
        default=None,
        help="Carpeta con la referencia congelada v4.1 (pNN.json). C8/C9 se miden contra ella y C3 usa la proporción bruta",
    )
    parser.add_argument(
        "--solo",
        default=None,
        help="Lista de piezas separadas por comas (p. ej. p68,p37,p41,p30); las demás se ignoran",
    )
    args = parser.parse_args(argv)
    args.solo = [x.strip() for x in args.solo.split(",") if x.strip()] if args.solo else None

    carpetas = []
    for grupo in args.jaulas:
        for item in grupo:
            for sub in item.split(","):
                sub = sub.strip()
                if sub and sub not in carpetas:
                    carpetas.append(sub)
    args.jaulas = carpetas
    return args


def main():
    t_inicio = time.time()
    args = parsear_argumentos()

    os.makedirs(args.salida, exist_ok=True)
    datos_ref = cargar_referencias_hd(args.ref, args.lineas, solo=args.solo, carpeta_v4=args.referencia_v4)
    modo_v4 = bool(args.referencia_v4)

    # Determinar todas las carpetas para la hoja de comparación:
    # siempre incluye datos/base_tripo y datos/base_qremeshify (si existen) más cualquier carpeta pasada en --jaulas
    carpetas_base = []
    for b in ["datos/base_tripo", "datos/base_qremeshify"]:
        if os.path.isdir(b):
            carpetas_base.append(b)

    todas_carpetas = []
    vistas_norm = set()
    for c in carpetas_base + args.jaulas:
        cn = os.path.normcase(os.path.abspath(c))
        if cn not in vistas_norm and os.path.isdir(c):
            vistas_norm.add(cn)
            todas_carpetas.append(c)

    jaulas_norm = {os.path.normcase(os.path.abspath(c)) for c in args.jaulas}
    un_solo_metodo = (len(args.jaulas) == 1)

    evaluaciones_por_norm = {}
    columnas_comp = []

    for carpeta in todas_carpetas:
        cn = os.path.normcase(os.path.abspath(carpeta))
        nombre_metodo = os.path.basename(os.path.normpath(carpeta))

        if un_solo_metodo and cn in jaulas_norm:
            dir_metodo = args.salida
        else:
            dir_metodo = os.path.join(args.salida, nombre_metodo)

        ev = evaluar_carpeta(carpeta, datos_ref, dir_metodo, args.crease_auto, modo_v4=modo_v4)
        evaluaciones_por_norm[cn] = (ev, dir_metodo)

        if dir_metodo != args.salida:
            with open(os.path.join(dir_metodo, "resultados.json"), "w", encoding="utf-8") as fh:
                json.dump(ev, fh, indent=2, ensure_ascii=False)
            if modo_v4:
                escribir_resumen_v4_md([ev], os.path.join(dir_metodo, "resumen_v4.md"), args)
            else:
                escribir_resumen_md([ev], os.path.join(dir_metodo, "resumen.md"))

        rp = ev["resumen_paneles"]
        subtitulo = (
            f"Caras: {rp.get('caras_media', 0):.0f} | "
            f"SubD->HD {rp.get('dist_subd_a_ref_media', 0):.2f}mm | "
            f"C8 cob {_fmt(rp.get('aristas_diseno_cobertura_media_pct'), 0)}% | "
            f"C9 plan p95 {_fmt(rp.get('planitud_p95_media'))}mm"
        )
        columnas_comp.append({
            "metodo": nombre_metodo,
            "titulo": nombre_metodo,
            "subtitulo": subtitulo,
            "carpeta_imgs": dir_metodo,
        })

    # Determinar la evaluación principal (prefiriendo la carpeta que no sea base_tripo / base_qremeshify si se pasaron varias)
    evaluaciones_pedidas = [
        evaluaciones_por_norm[os.path.normcase(os.path.abspath(c))][0]
        for c in args.jaulas
        if os.path.normcase(os.path.abspath(c)) in evaluaciones_por_norm
    ]
    todas_evaluaciones = [evaluaciones_por_norm[os.path.normcase(os.path.abspath(c))][0] for c in todas_carpetas]

    bases_norm = {os.path.normcase(os.path.abspath(b)) for b in carpetas_base}
    no_bases_pedidas = [
         c for c in args.jaulas if os.path.normcase(os.path.abspath(c)) not in bases_norm
    ]
    if no_bases_pedidas:
        cn_princ = os.path.normcase(os.path.abspath(no_bases_pedidas[-1]))
    else:
        cn_princ = os.path.normcase(os.path.abspath(args.jaulas[-1]))
    principal, dir_principal = evaluaciones_por_norm[cn_princ]

    # Si se pidieron varias carpetas, copiar en la carpeta raíz de salida las imágenes del frente
    # con prefijo de método y las imágenes de cerca (<pieza>_cerca.png) del método principal
    if not un_solo_metodo:
        for col in columnas_comp:
            m = col["metodo"]
            c_dir = col["carpeta_imgs"]
            for modo in ("alambre", "sombreado"):
                for cam_id, _, _ in CAMARAS:
                    src = os.path.join(c_dir, f"frente_{modo}_{cam_id}.png")
                    dst = os.path.join(args.salida, f"frente_{m}_{modo}_{cam_id}.png")
                    if os.path.exists(src):
                        shutil.copyfile(src, dst)
        if dir_principal != args.salida:
            for pid in principal["piezas"].keys():
                for suf in ("cerca", "alambre_cerca", "sombreado_cerca"):
                    src_c = os.path.join(dir_principal, f"{pid}_{suf}.png")
                    dst_c = os.path.join(args.salida, f"{pid}_{suf}.png")
                    if os.path.exists(src_c):
                        shutil.copyfile(src_c, dst_c)

    # Generar hojas de comparación para las 3 vistas
    for cam_id, _, _ in CAMARAS:
        ruta_comp_cam = os.path.join(args.salida, f"hoja_comparacion_{cam_id}.png")
        generar_hoja_comparacion(columnas_comp, ruta_comp_cam, cam_id=cam_id)

    ruta_hoja_principal = os.path.join(args.salida, "hoja_comparacion.png")
    ruta_comp_alias = os.path.join(args.salida, "comparacion.png")
    ruta_34 = os.path.join(args.salida, "hoja_comparacion_34.png")
    if os.path.exists(ruta_34):
        shutil.copyfile(ruta_34, ruta_hoja_principal)
        shutil.copyfile(ruta_34, ruta_comp_alias)

    # Generar hoja comparativa de cerca con luz rasante para todos los paneles
    paneles_ids = sorted(
        [pid for pid, ( _, t ) in NOMBRES_PIEZAS.items() if t == "panel" and pid in datos_ref]
    )
    generar_hoja_cerca_paneles(
        columnas_comp,
        paneles_ids,
        os.path.join(args.salida, "hoja_cerca_paneles.png"),
    )

    salida_json = {
        "carpeta_jaulas": principal["carpeta_jaulas"],
        "metodo": principal["metodo"],
        "carpeta_ref": args.ref,
        "carpeta_lineas": args.lineas,
        "crease_auto_grados": args.crease_auto,
        "resumen": principal["resumen"],
        "resumen_paneles": principal["resumen_paneles"],
        "piezas": principal["piezas"],
        "comparacion_metodos": {ev["metodo"]: ev for ev in todas_evaluaciones},
    }

    if modo_v4:
        salida_json["contrato"] = "referencia_v4.1"
        salida_json["carpeta_referencia_v4"] = args.referencia_v4
        salida_json["solo"] = args.solo
        ruta_json = os.path.join(args.salida, "resultados_v4.json")
        with open(ruta_json, "w", encoding="utf-8") as fh:
            json.dump(salida_json, fh, indent=2, ensure_ascii=False)
        escribir_resumen_v4_md(todas_evaluaciones, os.path.join(args.salida, "resumen_v4.md"), args)
        dt = time.time() - t_inicio
        print(f"\n=== EVALUACIÓN v4.1 COMPLETADA EN {dt:.2f}s ===")
        for ev in todas_evaluaciones:
            for pid, p in sorted(ev["piezas"].items()):
                c8, c9 = p["aristas_diseno"], p["planitud_regiones"]
                print(f"[{ev['metodo']}] {pid} caras={p['caras']} C3b={p['c3_bruta_pct']:.1f}% "
                      f"C5 {p['dist_subd_a_ref']['media']:.3f}/{p['dist_subd_a_ref']['p95']:.3f} inv {p['dist_ref_a_subd']['media']:.3f}/{p['dist_ref_a_subd']['p95']:.3f} "
                      f"C8 {c8['estado']} p95={_fmt(c8['dist_p95_mm'])} cob={_fmt(c8['cobertura_05mm_pct'],1)} "
                      f"C9 {c9['estado']} peor={_fmt(c9['p95_max_mm'])}")
        print(f"Resultados escritos en: {args.salida}")
        return

    ruta_json = os.path.join(args.salida, "resultados.json")
    with open(ruta_json, "w", encoding="utf-8") as fh:
        json.dump(salida_json, fh, indent=2, ensure_ascii=False)

    ruta_md = os.path.join(args.salida, "resumen.md")
    escribir_resumen_md(todas_evaluaciones, ruta_md)

    dt = time.time() - t_inicio
    print(f"\n=== EVALUACIÓN VUELTA 2 COMPLETADA EN {dt:.2f}s ===")
    for ev in todas_evaluaciones:
        rp = ev["resumen_paneles"]
        print(
            f"[{ev['metodo']}] paneles={rp['num_piezas']} | caras_media={rp['caras_media']} "
            f"| cuadros={rp['cuadros_pct_media']}% | prop>3.5={rp['proporcion_mayor_3_5_pct_media']}% "
            f"| polos>=6={rp['polos_interiores_6_mas_total']} "
            f"| SubD->HD media={rp['dist_subd_a_ref_media']:.3f}mm p95_med={rp['dist_subd_a_ref_p95_media']:.3f}mm "
            f"| C8_aristas p95_med={rp['aristas_diseno_p95_media']:.3f}mm cob={rp['aristas_diseno_cobertura_media_pct']:.1f}% ({rp['aristas_diseno_cumplen']}/{rp['num_piezas']}) "
            f"| C9_planitud p95_med={rp['planitud_p95_media']:.3f}mm p95_max_med={rp['planitud_p95_max_media']:.3f}mm ({rp['planitud_cumplen']}/{rp['num_piezas']}) "
            f"| cumplen_1_5={rp['piezas_cumplen_1_a_5']}/{rp['num_piezas']} "
            f"| cumplen_1_9={rp['piezas_cumplen_todo']}/{rp['num_piezas']}"
        )
    print(f"Resultados escritos en: {args.salida}")


if __name__ == "__main__":
    main()
