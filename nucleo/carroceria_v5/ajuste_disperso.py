"""Piloto de ajuste inverso SubD sin aumentar el numero de caras.

Se mide posteriormente con Blender; el ajuste no declara aprobacion C5/C8/C9.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
from scipy.sparse import coo_matrix, eye, vstack
from scipy.sparse.linalg import spsolve
import jaula_grande as base
j = base.j


def stencils(faces, crease, count, levels=2):
    matrix = eye(count, format='csr')
    faces = [list(f) for f in faces]
    edges = j.aristas(faces)
    tracked = {tuple(sorted(e)) for e in crease if len(edges.get(tuple(sorted(e)), [])) == 2}
    sharp = {tuple(sorted(e)) for e in crease}
    for _ in range(levels):
        edges = j.aristas(faces)
        sharp |= {e for e, fs in edges.items() if len(fs) == 1}
        n = matrix.shape[0]
        edge_ids = {e: n+i for i, e in enumerate(edges)}
        face_start = n+len(edges)
        adjacent, incident, sharp_neighbors = defaultdict(list), defaultdict(list), defaultdict(list)
        for fi, f in enumerate(faces):
            for a in f: incident[a].append(fi)
        for a, b in edges:
            adjacent[a].append(b); adjacent[b].append(a)
            if (a, b) in sharp:
                sharp_neighbors[a].append(b); sharp_neighbors[b].append(a)
        rows, cols, data = [], [], []
        def add(row, col, weight):
            rows.append(row); cols.append(col); data.append(weight)
        def face_weights(row, fi, weight):
            for x in faces[fi]: add(row, x, weight/len(faces[fi]))
        for a in range(n):
            cs, ns, fs = sharp_neighbors[a], adjacent[a], incident[a]
            if len(cs) > 2 or len(fs) <= 1 or not ns:
                add(a, a, 1.)
            elif len(cs) == 2:
                add(a, a, .75)
                for x in cs: add(a, x, .125)
            else:
                k = len(ns)
                add(a, a, (k-2)/k)
                for x in ns: add(a, x, 1/(k*k))
                for fi in fs: face_weights(a, fi, 1/(k*len(fs)))
        for e, fs in edges.items():
            row = edge_ids[e]
            if e in sharp:
                for a in e: add(row, a, .5)
            else:
                if len(fs) != 2: raise ValueError('Arista no manifold')
                for a in e: add(row, a, .25)
                for fi in fs: face_weights(row, fi, .25)
        for fi in range(len(faces)): face_weights(face_start+fi, fi, 1.)
        step = coo_matrix((data, (rows, cols)), shape=(face_start+len(faces), n)).tocsr()
        matrix = step @ matrix
        new_faces = []
        for fi, f in enumerate(faces):
            for k, a in enumerate(f):
                new_faces.append([a, edge_ids[tuple(sorted((a, f[(k+1)%len(f)])))],
                                  face_start+fi, edge_ids[tuple(sorted((a, f[k-1])))]] )
        tracked = {tuple(sorted((a, edge_ids[e]))) for e in tracked for a in e}
        sharp = {tuple(sorted((a, edge_ids[e]))) for e in sharp for a in e}
        faces = new_faces
    return matrix, faces, tracked


def closest_segments(points, segments, deadline):
    starts = segments[:, 0]
    directions = segments[:, 1]-starts
    lengths = np.maximum(np.sum(directions**2, axis=1), 1e-24)
    projections, indices, fractions = [], [], []
    for first in range(0, len(points), 128):
        if time.perf_counter() >= deadline: raise TimeoutError('Ajuste agotado')
        p = points[first:first+128]
        t = np.clip(np.sum((p[:, None]-starts)*directions, axis=2)/lengths, 0, 1)
        q = starts+t[:, :, None]*directions
        idx = np.argmin(np.sum((p[:, None]-q)**2, axis=2), axis=1)
        projections.extend(q[np.arange(len(p)), idx]); indices.extend(idx)
        fractions.extend(t[np.arange(len(p)), idx])
    return np.asarray(projections), np.asarray(indices), np.asarray(fractions)


def fit(vertices, faces, crease, reference_vertices, reference_faces, reference, seconds=120, steps=3,
        crease_weight=5., plane_weight=8.):
    if not all(np.isfinite(x) and x > 0 for x in (seconds, crease_weight, plane_weight)):
        raise ValueError('Presupuesto y pesos deben ser positivos y finitos')
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('Se requiere un numero positivo de iteraciones')
    if not np.isfinite(vertices).all() or not np.isfinite(reference_vertices).all():
        raise ValueError('Coordenadas no finitas')
    if not faces or any(len(face) != 4 or len(set(face)) != 4 for face in faces):
        raise ValueError('La candidata debe contener cuadros simples')
    edges = j.aristas(faces)
    if any(len(ids) > 2 for ids in edges.values()) or any(tuple(sorted(e)) not in edges for e in crease):
        raise ValueError('Topologia o crease invalido')
    started = time.perf_counter(); deadline = started+seconds
    original = np.asarray(vertices, dtype=float)
    vertices = original.copy()
    matrix, sub_faces, sub_crease = stencils(faces, crease, len(vertices), 2)
    tree = j.BVH(reference_vertices, reference_faces)
    segments = np.asarray([[a, b] for line in reference.get('lineas_doblez', [])
                           for a, b in zip(line['puntos'][:-1], line['puntos'][1:])])
    line_samples = np.concatenate([a+(b-a)*np.linspace(0, 1, max(2, int(np.ceil(np.linalg.norm(b-a)/2))+1))[:, None]
                                   for a, b in segments]) if len(segments) else np.empty((0, 3))
    edge_ids = np.asarray(sorted(sub_crease), dtype=int)
    line_ids = sorted({v for e in sub_crease for v in e})
    region_trees = [(region, j.BVH(reference_vertices, [reference_faces[i] for i in region['caras']]))
                    for region in reference.get('regiones_planas', [])]
    normals, _ = j.normales(vertices, faces)
    history = []; timed_out = False
    for iteration in range(steps):
        if time.perf_counter() >= deadline-3: break
        points = matrix @ vertices
        distances, targets = tree.distance(points, True)
        blocks, rhs = [matrix, .15*eye(len(vertices), format='csr')], [targets, .15*original]
        if len(segments) and len(line_ids):
            try:
                q, _, _ = closest_segments(points[line_ids], segments, deadline)
            except TimeoutError:
                timed_out = True
                break
            blocks.append(crease_weight*matrix[line_ids]); rhs.append(crease_weight*q)
            try:
                _, near, t = closest_segments(line_samples, points[edge_ids], deadline)
            except TimeoutError:
                timed_out = True
                break
            rows = matrix[edge_ids[near, 0]].multiply((1-t)[:, None])+matrix[edge_ids[near, 1]].multiply(t[:, None])
            blocks.append(crease_weight*rows); rhs.append(crease_weight*line_samples)
        for region, region_tree in region_trees:
            near = region_tree.distance(points)
            selected = np.flatnonzero(near <= distances+1e-5)
            if not len(selected): continue
            plane = np.asarray(region['plano'])
            corrected = points[selected]-(points[selected]@plane[:3]+plane[3])[:, None]*plane[:3]
            blocks.append(plane_weight*matrix[selected]); rhs.append(plane_weight*corrected)
        if time.perf_counter() >= deadline-3: break
        system = vstack(blocks, format='csr'); target = np.vstack(rhs)
        solved = spsolve((system.T @ system).tocsc(), system.T @ target)
        movement = solved-vertices
        # Limita desplazamientos y rechaza inversion/degeneracion, sin nuevos soportes.
        largest = float(np.linalg.norm(movement, axis=1).max())
        lengths = np.linalg.norm(movement, axis=1)
        movement *= np.minimum(1., 2./np.maximum(lengths, 1e-12))[:, None]
        for _ in range(24):
            candidate = np.asarray(vertices+movement, dtype=np.float32).astype(float)
            new_normals, areas = j.normales(candidate, faces)
            bad = (areas < 1e-7) | (np.sum(normals*new_normals, axis=1) <= 0)
            bad |= j.crossed_faces(candidate, faces, new_normals)
            if not np.any(bad): break
            blocked = sorted({vi for fi in np.flatnonzero(bad) for vi in faces[fi]})
            movement[blocked] *= .5
        old_residual = float(np.mean((system@vertices-target)**2))
        accepted = 0.
        for fraction in (1., .5, .25, .125, .0625):
            candidate = np.asarray(vertices+fraction*movement, dtype=np.float32).astype(float)
            new_normals, areas = j.normales(candidate, faces)
            if np.min(areas) < 1e-7 or np.any(np.sum(normals*new_normals, axis=1) <= 0): continue
            if np.any(j.crossed_faces(candidate, faces, new_normals)): continue
            new_residual = float(np.mean((system@candidate-target)**2))
            if new_residual < old_residual:
                vertices = candidate; accepted = fraction; break
        history.append({'iteracion': iteration, 'fraccion': accepted, 'max_paso_no_limitado_mm': largest,
                        'residuo_antes': old_residual, 'segundos': time.perf_counter()-started})
        if not accepted: break
    return vertices, {'historia': history, 'segundos': time.perf_counter()-started,
                      'nivel_modelo': 2, 'requiere_banco_blender': True,
                      'peso_crease': crease_weight, 'peso_plano': plane_weight,
                      'plazo_agotado': timed_out or time.perf_counter() >= deadline}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('entrada', type=Path); parser.add_argument('referencia_obj', type=Path)
    parser.add_argument('salida', type=Path); parser.add_argument('--segundos', type=float, default=120)
    parser.add_argument('--pasos', type=int, default=3)
    parser.add_argument('--peso-crease', type=float, default=5.)
    parser.add_argument('--peso-plano', type=float, default=8.)
    args = parser.parse_args()
    args.salida = base.validar_salida_experimental(args.entrada, args.salida, extra_inputs=(args.referencia_obj,))
    frozen = base.frozen_reference(args.referencia_obj)
    if frozen is None: raise ValueError('Referencia congelada requerida')
    v, f = j.leer_obj(args.entrada); rv, rf = j.leer_obj(args.referencia_obj)
    cs = json.loads(args.entrada.with_name(args.entrada.stem+'_crease.json').read_text())['crease']
    output, info = fit(v, f, cs, rv, rf, json.loads(frozen.read_text()), args.segundos, args.pasos,
                       args.peso_crease, args.peso_plano)
    info.update(metodo='ajuste inverso disperso sin caras nuevas', metricas=base.topology(output, f),
                sha256_entrada=base.sha(args.entrada), sha256_referencia=base.sha(frozen), aceptado=False)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    j.escribir(args.salida, output, f, cs, info, precision=12)
    print(json.dumps(info))


if __name__ == '__main__': main()
