"""Ajuste inverso Catmull-Clark con correspondencias en ambos sentidos."""
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from scipy.sparse import eye, vstack
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'carroceria_v5'))
from ajuste_disperso import stencils, closest_segments, j
from superficie import Surface


def _ratios(vertices, faces):
    p = vertices[np.asarray(faces)]
    lengths = np.linalg.norm(p-np.roll(p, -1, axis=1), axis=2)
    return lengths.max(axis=1)/np.maximum(lengths.min(axis=1), 1e-12)


def _samples(surface, maximum=1800):
    weights = np.asarray([(a/3, b/3, 1-(a+b)/3)
                          for a in range(4) for b in range(4-a)])
    samples = np.einsum('ki,tij->tkj', weights, surface.triangles).reshape(-1, 3)
    if len(samples) > maximum:
        samples = samples[np.linspace(0, len(samples)-1, maximum).astype(int)]
    return samples


def _guide_vertices(faces, crease, count, levels):
    # El stencil reutilizado sólo rastrea crease interiores; aquí también cuentan
    # las guías declaradas que coinciden con el borde abierto.
    tracked = {tuple(sorted(e)) for e in crease}
    for _ in range(levels):
        edges = j.aristas(faces)
        edge_ids = {edge: count+i for i, edge in enumerate(edges)}
        center_start = count+len(edges)
        count = center_start+len(faces)
        tracked = {tuple(sorted((a, edge_ids[edge]))) for edge in tracked for a in edge}
        faces = [[a, edge_ids[tuple(sorted((a, face[(k+1) % len(face)])))],
                  center_start+i, edge_ids[tuple(sorted((a, face[k-1])))]]
                 for i, face in enumerate(faces) for k, a in enumerate(face)]
    return sorted({a for edge in tracked for a in edge})


def fit(vertices, faces, crease, source_vertices, source_faces, guide_lines,
        seam_vertices=(), seconds=100, steps=4):
    if not np.isfinite(seconds) or seconds <= 0 or not isinstance(steps, int) or steps < 1:
        raise ValueError('Presupuesto e iteraciones deben ser positivos')
    started = time.perf_counter()
    deadline = started+seconds
    original = np.asarray(vertices, float)
    vertices = original.copy()
    seam = np.asarray(sorted(set(seam_vertices)), int)
    if len(seam):
        vertices[seam, 0] = 0.
    source = Surface(source_vertices, source_faces)
    reference_samples = _samples(source)
    matrix, subdiv_faces, _ = stencils(faces, crease, len(vertices), 2)
    guide_segments = np.asarray([[a, b] for line in guide_lines
                                for a, b in zip(line[:-1], line[1:])
                                if np.linalg.norm(np.asarray(a)-b) > 1e-10])
    guide_vertices = _guide_vertices(faces, crease, len(vertices), 2)
    normals, _ = j.normales(vertices, faces)
    red_limit = int(np.count_nonzero(_ratios(vertices, faces) > 3.5))
    identity = eye(len(vertices), format='csr')
    history = []
    for iteration in range(steps):
        if time.perf_counter() >= deadline-5:
            break
        subdiv = matrix @ vertices
        targets = source.project(subdiv)
        reverse = Surface(subdiv, subdiv_faces)
        near, triangle_ids, barycentric = reverse.closest(reference_samples)
        tri = reverse.faces[triangle_ids]
        reverse_rows = sum(matrix[tri[:, k]].multiply(barycentric[:, k, None])
                           for k in range(3))
        # Normalizar evita que la densidad de un muestreo domine al otro.
        forward_weight = 1/np.sqrt(len(subdiv))
        reverse_weight = 1/np.sqrt(len(reference_samples))
        blocks = [forward_weight*matrix, reverse_weight*reverse_rows,
                  .03/np.sqrt(len(vertices))*identity]
        rhs = [forward_weight*targets, reverse_weight*reference_samples,
               .03/np.sqrt(len(vertices))*original]
        if len(guide_segments) and guide_vertices:
            try:
                projected, _, _ = closest_segments(subdiv[guide_vertices], guide_segments, deadline)
            except TimeoutError:
                break
            weight = 2/np.sqrt(len(guide_vertices))
            blocks.append(weight*matrix[guide_vertices])
            rhs.append(weight*projected)
        system = vstack(blocks, format='csr')
        target = np.vstack(rhs)
        if time.perf_counter() >= deadline-3:
            break
        solved = spsolve((system.T @ system).tocsc(), system.T @ target)
        if not np.isfinite(solved).all():
            raise ValueError('Ajuste inverso singular o no finito')
        if len(seam):
            solved[seam, 0] = 0.
        movement = solved-vertices
        distance = np.linalg.norm(movement, axis=1)
        movement *= np.minimum(1., 2/np.maximum(distance, 1e-12))[:, None]
        old_residual = float(np.sum((system@vertices-target)**2))
        fraction_accepted = 0.
        for fraction in (1., .5, .25, .125, .0625):
            candidate = vertices+fraction*movement
            if len(seam):
                candidate[seam, 0] = 0.
            candidate_normals, areas = j.normales(candidate, faces)
            if np.min(areas) < 1e-8 or np.any(np.sum(normals*candidate_normals, axis=1) <= 0):
                continue
            if np.any(j.crossed_faces(candidate, faces, candidate_normals)):
                continue
            if np.count_nonzero(_ratios(candidate, faces) > 3.5) > red_limit:
                continue
            residual = float(np.sum((system@candidate-target)**2))
            if residual < old_residual:
                vertices = candidate
                fraction_accepted = fraction
                break
        history.append({'iteracion': iteration, 'fraccion': fraction_accepted,
                        'objetivo_antes': old_residual,
                        'media_directa_antes_mm': float(np.linalg.norm(subdiv-targets, axis=1).mean()),
                        'media_inversa_antes_mm': float(np.linalg.norm(reference_samples-near, axis=1).mean())})
        if not fraction_accepted:
            break
    return vertices, {'metodo': 'minimos_cuadrados_SubD_bidireccional', 'historia': history,
                      'segundos': time.perf_counter()-started, 'nivel_modelo': 2,
                      'costura_x_cero': bool(not len(seam) or np.all(vertices[seam, 0] == 0)),
                      'requiere_banco_blender': True}
