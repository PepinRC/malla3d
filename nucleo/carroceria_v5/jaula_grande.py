"""Convierte un OBJ Smart mixto en una jaula de cuadros sin recetas por nombre.

Uso: python nucleo/carroceria_v5/jaula_grande.py entrada.obj salida.obj
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import subprocess
import tempfile
from importlib.metadata import PackageNotFoundError, version

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Solo lectura: estas primitivas ya manejan OBJ, orientación y bandas conformes.
g = _module('carroceria_lowpoly', ROOT / 'nucleo' / 'lowpoly' / 'generar.py')
j = g.j


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validar_salida_experimental(source, output, *, json_result=False, extra_inputs=()):
    """Los pilotos no pueden publicar en principal ni reemplazar sus entradas."""
    output = Path(output).resolve()
    roots = [ROOT/'salidas/carroceria_v5/variantes']
    if json_result:
        roots += [ROOT/'evaluacion/resultados_v5/carroceria_ext',
                  ROOT/'evaluacion/resultados_v5/carroceria_ext_iter']
    if output.suffix.lower() != ('.json' if json_result else '.obj'):
        raise ValueError('Extension de salida experimental incorrecta')
    protected = set()
    for path in (source, *extra_inputs):
        path = Path(path).resolve()
        protected.update((path, path.with_name(path.stem+'_crease.json').resolve(),
                          path.with_name(path.stem+'_info.json').resolve()))
    targets = [output] if json_result else [output, output.with_name(output.stem+'_crease.json').resolve(),
                                           output.with_name(output.stem+'_info.json').resolve()]
    if any(path in protected or not any(path.is_relative_to(root.resolve()) for root in roots) for path in targets):
        raise ValueError('Salida experimental fuera de variantes/resultados o coincide con una entrada')
    return output


def scipy_version():
    try:
        return version('scipy')
    except PackageNotFoundError:
        return None


def frozen_reference(source):
    """Identifica una referencia congelada por contenido, nunca por el nombre."""
    digest = sha(source)
    for path in sorted((ROOT / 'evaluacion' / 'referencia_smart_v5').glob('s*.json')):
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if data.get('sha256_hd') == digest:
            return path
    return None


def convex_quad(v, quad, normal):
    p = v[quad]
    turns = np.cross(np.roll(p, -1, axis=0) - p,
                     np.roll(p, -2, axis=0) - np.roll(p, -1, axis=0))
    return bool(np.all(turns @ normal > 1e-10))


def triangle_candidates(v, faces, frozen):
    """Aristas tri-tri aptas; no atraviesa líneas de doblez congeladas."""
    edges = j.aristas(faces)
    normals, _ = j.normales(v, faces)
    protected = set(g.design_crease(v, faces, frozen)) if frozen else set()
    out = []
    for edge, ids in edges.items():
        if edge in protected or len(ids) != 2 or any(len(faces[i]) != 3 for i in ids):
            continue
        a, b = ids
        quad = j.unir_contornos(faces[a], faces[b])
        if quad is None or len(quad) != 4:
            continue
        dot = float(normals[a] @ normals[b])
        if dot < math.cos(math.radians(25)) or not convex_quad(v, quad, normals[a] + normals[b]):
            continue
        aspect = j.aspecto(v, quad)
        if aspect > 8:
            continue
        # Prioriza caras cuadradas y coplanares; el sesgo fijo evita pares marginales.
        out.append((aspect + 2.0 * (1.0 - dot), a, b, quad))
    return out, protected


def global_triangle_matching(v, faces, frozen, limit_seconds=25):
    """Matching de peso mínimo. scipy MILP es opcional para mantener el CLI portable."""
    candidates, protected = triangle_candidates(v, faces, frozen)
    selected = []
    solver = 'greedy'
    if candidates:
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import csc_array
            triangles = sorted({x for _, a, b, _ in candidates for x in (a, b)})
            row = {face: i for i, face in enumerate(triangles)}
            rows = [row[x] for _, a, b, _ in candidates for x in (a, b)]
            cols = np.repeat(np.arange(len(candidates)), 2)
            matrix = csc_array((np.ones(len(rows)), (rows, cols)), shape=(len(row), len(candidates)))
            # Esta cota domina la suma de calidad de cualquier matching: primero
            # máxima cardinalidad, después convexidad/aspecto.
            bonus = 11.0 * len(candidates)
            cost = np.array([score - bonus for score, _, _, _ in candidates])
            result = milp(c=cost, integrality=np.ones(len(candidates)), bounds=Bounds(0, 1),
                          constraints=LinearConstraint(matrix, -np.inf, 1),
                          options={'time_limit': limit_seconds, 'mip_rel_gap': 0.0})
            if result.x is not None:
                selected = [candidates[i] for i, x in enumerate(result.x) if x > .5]
                solver = 'scipy.optimize.milp:' + str(result.status)
        except (ImportError, ValueError, RuntimeError):
            pass
    if not selected:
        used = set()
        for candidate in sorted(candidates):
            _, a, b, _ = candidate
            if a not in used and b not in used:
                selected.append(candidate)
                used.update((a, b))
    taken = {face for _, a, b, _ in selected for face in (a, b)}
    output = [face for index, face in enumerate(faces) if index not in taken]
    output.extend(quad for _, _, _, quad in selected)
    removed = []
    for _, a, b, _ in selected:
        common = set(zip(faces[a], faces[a][1:] + faces[a][:1])) | set(zip(faces[a][1:] + faces[a][:1], faces[a]))
        other = set(zip(faces[b], faces[b][1:] + faces[b][:1])) | set(zip(faces[b][1:] + faces[b][:1], faces[b]))
        removed.append(list(min(tuple(sorted(edge)) for edge in common & other)))
    assert not set(map(tuple, removed)) & protected
    return output, dict(candidatos=len(candidates), pares=len(selected), protegidas=len(protected),
                        aristas_eliminadas=removed, solucionador=solver)


def quadrangulate_remainder(v, faces, deadline):
    """Convierte los triángulos restantes por bandas, sin repetir matching."""
    history = []
    while any(len(face) == 3 for face in faces):
        if time.perf_counter() >= deadline:
            raise TimeoutError('Presupuesto de 300 s agotado durante cuadrangulación')
        triangle = next(i for i, face in enumerate(faces) if len(face) == 3)
        candidates = []
        for a, b in zip(faces[triangle], faces[triangle][1:] + faces[triangle][:1]):
            selected = g.strip(faces, (a, b))
            try:
                vv, ff = g.split_strip(v, faces, selected)
            except ValueError:
                continue
            ratios = np.array([j.aspecto(vv, quad) for quad in ff])
            candidates.append((float(np.sum(np.maximum(ratios - 3.5, 0)) + .15 * len(ff)), vv, ff, len(selected)))
        if not candidates:
            raise ValueError('No se encontró banda conforme para un triángulo')
        _, v, faces, count = min(candidates, key=lambda item: item[0])
        history.append(count)
    return v, faces, history


def pole_ring(faces, vertex):
    """Devuelve caras [a,u,w,t] consecutivas alrededor de una estrella de quads."""
    wedges = {}
    for index, face in enumerate(faces):
        if vertex not in face or len(face) != 4:
            continue
        at = face.index(vertex)
        q = face[at:] + face[:at]
        if q[1] in wedges:
            return None
        wedges[q[1]] = (index, q)
    if len(wedges) < 6:
        return None
    ring = []
    count = len(wedges)
    key = min(wedges)
    while key in wedges and len(ring) < count:
        index, wedge = wedges.pop(key)
        ring.append((index, wedge))
        key = wedge[3]
    return ring if not wedges and key == ring[0][1][1] else None


def split_pole(v, faces, vertex, deadline, degree=None, border=None):
    """Parte una estrella 6+ en dos polos; V+1/E+2/F+1 preserva Euler."""
    ring = pole_ring(faces, vertex)
    if ring is None or time.perf_counter() >= deadline:
        return None
    n = len(ring)
    k = n // 2
    old_normals, _ = j.normales(v, faces)
    best = None
    for shift in range(n):
        ordered = ring[shift:] + ring[:shift]
        group = ordered[:k]
        start, end = group[0][1][1], group[-1][1][3]
        if degree is not None:
            # Se permiten extremos de grado cinco solo si el exceso global de
            # valencia disminuye; así un arreglo no desplaza el polo a un vecino.
            old_excess = max(n - 5, 0)
            new_excess = max(n - k + 1 - 5, 0) + max(k + 1 - 5, 0)
            for endpoint in (start, end):
                if endpoint not in border:
                    new_excess += max(degree[endpoint] + 1 - 5, 0) - max(degree[endpoint] - 5, 0)
            if new_excess >= old_excess:
                continue
        chord = v[end] - v[start]
        chord2 = float(chord @ chord)
        if chord2 < 1e-12:
            continue
        q = v[start] + chord * np.clip(float((v[vertex] - v[start]) @ chord) / chord2, 0, 1)
        local_normal = sum(old_normals[index] for index, _ in ordered)
        local_normal /= max(float(np.linalg.norm(local_normal)), 1e-20)
        width = np.cross(local_normal, chord)
        width /= max(float(np.linalg.norm(width)), 1e-20)
        group_points = np.array([v[x] for _, wedge in group for x in wedge[1:]])
        side = 1 if float((group_points.mean(axis=0) - q) @ width) >= 0 else -1
        # Desplazamiento pequeño en el plano local: evita el puente degenerado.
        for height in (.05, .15, .30, .60):
            if time.perf_counter() >= deadline:
                return best
            vv = v.copy()
            outside = q - side * width * height
            inside = q + side * width * height
            if max(np.linalg.norm(outside - v[vertex]), np.linalg.norm(inside - v[vertex])) > 1.5:
                continue
            delta = outside - v[vertex]
            vv[vertex] = outside
            duplicate = len(vv)
            vv = np.vstack((vv, inside))
            selected = {index for index, _ in group}
            ff = [([duplicate if x == vertex else x for x in face] if i in selected else face[:])
                  for i, face in enumerate(faces)]
            bridge = [vertex, start, duplicate, end]
            ff.append(bridge)
            local_ids = sorted(selected | {index for index, _ in ordered[k:]} | {len(ff) - 1})
            local = [ff[i] for i in local_ids]
            normals, areas = j.normales(vv, local)
            if (np.min(areas) < 1e-7 or any(j.crossed_faces(vv, local, normals))
                    or not convex_quad(vv, bridge, normals[-1])):
                continue
            # El parche no puede invertir el sentido de las caras que existían.
            prior = np.array([old_normals[i] if i < len(faces) else normals[-1] for i in local_ids])
            if np.any(np.sum(normals * prior, axis=1) <= 1e-8):
                continue
            score = sum(j.aspecto(vv, face) for face in local) + 5 * max(j.aspecto(vv, bridge) - 3.5, 0)
            if best is None or score < best[0]:
                best = (score, vv, ff, dict(polo=vertex, valencia=n, caras_grupo=k,
                                              desplazamiento_mm=float(np.linalg.norm(delta)), inicio=start, fin=end))
    return best


def components(faces):
    edges = j.aristas(faces)
    adjacent = defaultdict(set)
    for ids in edges.values():
        for a in ids:
            adjacent[a].update(ids)
    remaining = set(range(len(faces))); count = 0
    while remaining:
        count += 1; todo = [remaining.pop()]
        while todo:
            todo.extend(x for x in adjacent[todo.pop()] if x in remaining)
            remaining.difference_update(todo)
    return count


def topology(v, faces):
    result = g.topology(v, faces)
    result['componentes'] = components(faces)
    result['triangulos'] = sum(len(face) == 3 for face in faces)
    result['caras_no_cuadro'] = sum(len(face) != 4 for face in faces)
    return result


def support_stage(v, faces, crease, reference, output, deadline):
    """Una etapa Blender aislada; el generador conserva su presupuesto total."""
    available = min(70., deadline-time.perf_counter()-120.)
    if reference is None or available <= 1:
        return v, faces, crease, {'omitido': 'sin referencia congelada o presupuesto disponible'}
    blender = Path('C:/Program Files/Blender Foundation/Blender 5.2/blender.exe')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.soporte-', dir=output.parent) as folder:
        before, after = Path(folder)/'entrada.obj', Path(folder)/'salida.obj'
        j.escribir(before, np.asarray(v, dtype=np.float32).astype(float), faces, sorted(crease), {}, precision=12)
        command = [str(blender), '-b', '--factory-startup', '--python-exit-code', '1',
                   '-P', str(HERE/'aplicar_soportes.py'), '--', str(before), str(reference), str(after),
                   '--deadline-seconds', str(max(1., available-5.))]
        try:
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                       encoding='utf-8', errors='replace', timeout=available, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            detail = str(error.stderr or error.stdout or error)[-3000:]
            return v, faces, crease, {'fallo': detail, 'omitido': 'se conserva la geometría previa válida'}
        vv, ff = j.leer_obj(after)
        cc = {tuple(pair) for pair in json.loads(after.with_name('salida_crease.json').read_text())['crease']}
        audit = json.loads(after.with_name('salida_info.json').read_text())
        audit['entrada'] = 'etapa interna del generador'
        audit['blender_exit_code'] = completed.returncode
        return vv, ff, cc, audit


def build(source, output, milp_limit=25):
    started = time.perf_counter()
    deadline = started + 295.0
    source, output = source.resolve(), output.resolve()
    allowed = (ROOT / 'salidas' / 'carroceria_v5').resolve()
    if source.suffix.lower() != '.obj' or output.suffix.lower() != '.obj' or source == output:
        raise ValueError('Se requieren OBJ de entrada y salida distintos')
    if not source.is_relative_to(ROOT) or not output.is_relative_to(allowed):
        raise ValueError('La entrada debe estar en la raíz del proyecto y la salida en salidas/carroceria_v5')
    raw_v, raw_f = j.leer_obj(source)
    if (len(raw_v) == 0 or not raw_f or not np.isfinite(raw_v).all()
            or any(len(face) not in (3, 4) or len(set(face)) != len(face)
                   or min(face) < 0 or max(face) >= len(raw_v) for face in raw_f)):
        raise ValueError('La entrada requiere triángulos/cuadros simples')
    reference = frozen_reference(source)
    overlap = _module('carroceria_solapes', HERE / 'reparar_solapes.py')
    cleaned, overlaps = overlap.reparar(raw_v, raw_f, j, min(deadline, time.perf_counter()+20))
    v, faces, detached = g.separate_sheets(raw_v, cleaned)
    faces = g.orient(faces)
    topology_signature = g.signature(faces)
    sheets_before = components(raw_f)
    sheets_after = components(faces)
    faces, matching = global_triangle_matching(v, faces, reference, min(milp_limit, max(1, deadline - time.perf_counter())))
    v, faces, strips = quadrangulate_remainder(v, faces, deadline)
    sharp = set(g.design_crease(v, faces, reference))
    asymmetric = _module('carroceria_asymmetric', HERE / 'asymmetric_poles.py')
    v, faces, local_poles = asymmetric.repair(v, faces, sharp, min(deadline, time.perf_counter()+45))
    refiner = _module('carroceria_refinar', HERE/'refinar_cuadros.py')
    simpler = _module('carroceria_simplificar', HERE/'simplificar_cuadros.py')
    budget = max(0., min(60., deadline-time.perf_counter()-190.))
    v, faces, crease, refinement, _ = refiner.refine(v, faces, sharp, budget)
    faces, rotations, _ = simpler.rotate(v, faces, crease, max(0., min(10., deadline-time.perf_counter()-170.)))
    faces, coarsening, _ = simpler.coarsen(v, faces, crease, max(0., min(25., deadline-time.perf_counter()-145.)))
    v, faces, crease, supports = support_stage(v, faces, crease, reference, output, deadline)
    v, faces, pre_poles = asymmetric.repair(v, faces, crease, min(deadline-30., time.perf_counter()+25.))
    repairer = _module('carroceria_reparar_polos', HERE/'reparar_polos.py')
    v, faces, crease, pole_repairs = repairer.repair_batched(v, faces, crease, deadline-30.)
    # Se conserva el cero de polos; el refinador no permite crear valencias 6+.
    v, faces, crease, final_refinement, _ = refiner.refine(v, faces, crease, max(0., min(20., deadline-time.perf_counter()-5.)))
    protected_simplification = {'omitido': 'sin referencia congelada o presupuesto disponible'}
    if reference is not None and deadline-time.perf_counter() > 25:
        protected = _module('carroceria_simplificar_protegido', HERE/'simplificar_protegido.py')
        locked = protected.protected_vertices(v, faces, raw_v, raw_f, json.loads(reference.read_text()))
        budget = min(50., deadline-time.perf_counter()-8.)
        if budget > 1:
            v, faces, crease, protected_simplification = protected.simplify(v, faces, crease, locked, budget)
    faces = g.orient(faces)
    used = sorted({x for face in faces for x in face})
    remap = {old: new for new, old in enumerate(used)}
    v = v[used]
    faces = [[remap[x] for x in face] for face in faces]
    # El roundtrip float32 conserva las claves crease usadas por el banco Blender.
    v = np.asarray(v, dtype=np.float32).astype(float)
    _, exported_areas = j.normales(v, faces)
    crossed = j.crossed_faces(v, faces)
    if np.min(exported_areas) < 1e-7 or any(crossed):
        raise ValueError(f'Geometría final inválida tras float32: área mínima={exported_areas.min():.12g}, '
                         f'caras menores de 1e-7={np.flatnonzero(exported_areas < 1e-7)[:10].tolist()}, '
                         f'cruces={np.flatnonzero(crossed)[:10].tolist()}')
    final_edges = j.aristas(faces)
    crease = sorted({tuple(sorted((remap[a], remap[b]))) for a, b in crease
                     if a in remap and b in remap and len(final_edges.get(tuple(sorted((remap[a], remap[b]))), ())) == 2})
    info = dict(entrada=str(source.relative_to(ROOT)), sha256_entrada=sha(source), sha256_codigo=sha(__file__),
                dependencias=dict(numpy=np.__version__, scipy=scipy_version()),
                referencia_diseno=(str(reference.relative_to(ROOT)) if reference else None),
                referencia_por_sha=bool(reference), metodo='reparación Smart, soporte selectivo, reparación de polos y simplificación protegida por planos fijos',
                matching=matching, bandas_divididas=len(strips), solapes_retirados=overlaps,
                divisiones_locales_polos=local_poles,
                refinamiento=refinement, rotaciones=rotations, simplificacion=coarsening,
                soportes=supports, pre_reparacion_polos=pre_poles, reparacion_polos=pole_repairs,
                refinamiento_final=final_refinement,
                simplificacion_protegida=protected_simplification,
                sha256_modulos={p.name: sha(p) for p in [HERE/name for name in (
                    'reparar_solapes.py', 'asymmetric_poles.py', 'reparar_polos.py', 'geometria_polos.py',
                    'refinar_cuadros.py', 'simplificar_cuadros.py', 'aplicar_soportes.py',
                    'soporte_crease.py', 'soporte_planos.py', 'medir_soporte.py',
                    'simplificar_protegido.py')]},
                sha256_dependencias_lectura={name: sha(ROOT/'nucleo/lowpoly'/name)
                    for name in ('generar.py', 'jaula_base.py', 'p37_v2.py')},
                continuidad=dict(componentes_entrada=sheets_before, componentes_tras_separar=sheets_after,
                                 firma_euler_tras_separar=topology_signature, firma_euler_final=g.signature(faces),
                                 caras_separadas_no_manifold=detached, caras_separadas_final=0,
                                 advertencia=('separate_sheets separó hojas; revisar continuidad visual' if detached or sheets_after != sheets_before else None)),
                metricas=topology(v, faces), aceptado=False,
                advertencias=['C5, C8 y C9 requieren el banco independiente; no se declara aceptación por generar el OBJ.'] +
                              (['Las hojas no manifold restantes se separaron; revisar continuidad.'] if detached else []),
                segundos=time.perf_counter() - started)
    if time.perf_counter() >= deadline:
        raise TimeoutError('Presupuesto de 300 s agotado antes de escribir salida')
    j.escribir(output, v, faces, crease, info, precision=12)
    print(json.dumps(dict(salida=str(output), **info['metricas'], segundos=info['segundos']), ensure_ascii=False), flush=True)
    return info


def self_test():
    v = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]])
    faces = [[0, 1, 2], [0, 2, 3], [1, 4, 5, 2]]
    output, info = global_triangle_matching(v, faces, None, 2)
    assert info['pares'] == 1 and all(len(face) == 4 for face in output)
    assert topology(v, output)['polos_interiores_6_mas'] == 0
    print('SELFTEST PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('entrada', type=Path, nargs='?')
    parser.add_argument('salida', type=Path, nargs='?')
    parser.add_argument('--milp-segundos', type=float, default=25)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.entrada and args.salida:
        build(args.entrada, args.salida, args.milp_segundos)
    else:
        parser.error('indique entrada.obj salida.obj')


if __name__ == '__main__':
    main()

