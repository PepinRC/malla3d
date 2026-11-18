"""Primitivas estrechas de medicion para soportes Blender, sin renders."""
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
from pathlib import Path
import time
import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parents[2]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

banco = load('support_exact_bank', ROOT/'evaluacion/evaluar_blender.py')
j = load('support_geometry', ROOT/'nucleo/lowpoly/jaula_base.py')

def check_deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError('Plazo total de soporte agotado; no se exporta candidato parcial')

def frozen_reference(path):
    path = Path(path).resolve()
    reference = json.loads(path.read_text(encoding='utf-8-sig'))
    if reference.get('contrato') != 'referencia_v4.1' or not reference.get('fuente'):
        raise ValueError('Se requiere referencia_v4.1 con fuente congelada')
    source = Path(reference['fuente'])
    source = source if source.is_absolute() else ROOT/source
    vertices, _ = banco.leer_obj(source)
    parsed = banco.cargar_referencia_v4(str(path.parent), path.stem, str(source), vertices)
    if not parsed or not parsed['sha_coincide']:
        raise ValueError('SHA de fuente congelada no coincide')
    if any(plane['plano'].shape != (4,) or not np.isfinite(plane['plano']).all()
           or np.linalg.norm(plane['plano'][:3]) < 1e-8 for plane in parsed['planos']):
        raise ValueError('Plano congelado invalido')
    return reference, parsed

def geometry_guard(bm, faces=None):
    bm.verts.index_update()
    vertices = np.array([tuple(vertex.co) for vertex in bm.verts])
    faces = [[vertex.index for vertex in face.verts] for face in (bm.faces if faces is None else faces) if face.is_valid]
    if any(len(face) != 4 for face in faces): return False
    if not faces: return True
    normals, areas = j.normales(vertices, faces)
    return bool(np.isfinite(vertices).all() and min(areas) >= 1e-7 and not any(j.crossed_faces(vertices, faces, normals)))

def raw_bmesh(vertices, faces, crease):
    """Sin soldadura: conserva el contrato de indices del archivo de entrada."""
    vertices = np.asarray(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Vertices invalidos')
    if any(len(face) != 4 or len(set(face)) != 4 or min(face) < 0 or max(face) >= len(vertices) for face in faces):
        raise ValueError('Se requieren cuadros simples con indices validos')
    bm = bmesh.new()
    try:
        for point in vertices: bm.verts.new(point)
        bm.verts.ensure_lookup_table()
        for face in faces: bm.faces.new([bm.verts[index] for index in face])
        bm.normal_update(); bm.verts.index_update()
        layer = bm.edges.layers.float.new('crease_edge')
        edges = {tuple(sorted(vertex.index for vertex in edge.verts)): edge for edge in bm.edges}
        for pair in crease:
            key = tuple(sorted(pair))
            if len(key) != 2 or key not in edges: raise ValueError('Pareja crease inexistente')
            edges[key][layer] = 1.
        return bm
    except Exception:
        bm.free(); raise

def arrays_from_bm(bm):
    bm.verts.index_update()
    layer = bm.edges.layers.float['crease_edge']
    return (np.array([tuple(vertex.co) for vertex in bm.verts]),
            [[vertex.index for vertex in face.verts] for face in bm.faces],
            {tuple(sorted(vertex.index for vertex in edge.verts)) for edge in bm.edges if edge[layer] > .5})

def object_from_bm(bm):
    mesh = bpy.data.meshes.new('support_measure')
    bm.to_mesh(mesh); mesh.update()
    obj = bpy.data.objects.new('support_measure', mesh)
    bpy.context.collection.objects.link(obj)
    mod = obj.modifiers.new('Subd', 'SUBSURF')
    mod.subdivision_type = 'CATMULL_CLARK'; mod.levels = mod.render_levels = 3
    mod.boundary_smooth = 'PRESERVE_CORNERS'
    return obj

def remove_object(obj):
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True); bpy.data.meshes.remove(mesh)

class Segments:
    """BVH genera candidatos; distancia final exacta a segmentos originales."""
    def __init__(self, a, b):
        self.a, self.b = np.asarray(a), np.asarray(b)
        if not len(self.a): raise ValueError('No hay aristas crease interiores evaluables')
        self.d = self.b-self.a
        self.len2 = np.maximum(np.sum(self.d*self.d, axis=1), 1e-24)
        axis = np.zeros_like(self.d)
        axis[np.arange(len(axis)), np.argmin(np.abs(self.d), axis=1)] = 1.
        normal = np.cross(self.d, axis)
        normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-24)[:, None]
        triangles = np.stack([self.a, self.b, self.a+.001*normal], axis=1)
        self.tree = BVHTree.FromPolygons(triangles.reshape(-1, 3).tolist(), np.arange(len(a)*3).reshape(-1, 3).tolist(), all_triangles=True)

    def closest(self, points, deadline=None):
        distances, ids = np.empty(len(points)), np.empty(len(points), dtype=int)
        for i, point in enumerate(points):
            if i % 512 == 0: check_deadline(deadline)
            nearest = self.tree.find_nearest(Vector(point))
            indices = np.array([hit[2] for hit in self.tree.find_nearest_range(Vector(point), nearest[3]+.002)], dtype=int)
            t = np.clip(np.sum((point-self.a[indices])*self.d[indices], axis=1)/self.len2[indices], 0., 1.)
            ds = np.linalg.norm(point-self.a[indices]-t[:, None]*self.d[indices], axis=1)
            k = ds.argmin(); distances[i], ids[i] = ds[k], indices[k]
        return distances, ids

def extract_subd(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        vertices = np.empty(len(mesh.vertices)*3); mesh.vertices.foreach_get('co', vertices)
        vertices = vertices.reshape(-1, 3)
        edges = np.empty(len(mesh.edges)*2, dtype=int); mesh.edges.foreach_get('vertices', edges)
        edges = edges.reshape(-1, 2)
        loops = np.empty(len(mesh.loops), dtype=int); mesh.loops.foreach_get('edge_index', loops)
        counts = np.bincount(loops, minlength=len(edges))
        crease = np.array([item.value for item in mesh.attributes['crease_edge'].data])
        selected = (crease > .5) & (counts == 2)
        return Segments(vertices[edges[selected, 0]], vertices[edges[selected, 1]])
    finally:
        evaluated.to_mesh_clear()

def stats(distances, weights):
    return {'media': float(np.average(distances, weights=weights)),
            'p95': float(banco.percentil_ponderado(distances, weights, 95)),
            'max': float(distances.max()), 'coverage_05_pct': float(100*np.sum(weights[distances <= .5])/weights.sum())}

def measure_c8(vertices, faces, crease, points, weights, deadline=None):
    check_deadline(deadline)
    # Reproduce OBJ float32 roundtrip para las claves de remapeo crease del banco.
    vertices = np.asarray(vertices, dtype=np.float32).astype(float).tolist()
    bm, _ = banco.construir_bmesh_jaula(vertices, faces, list(crease), True, 30.)
    try: obj = object_from_bm(bm)
    finally: bm.free()
    try:
        distances, _ = extract_subd(obj).closest(points, deadline)
        return stats(distances, weights), distances
    finally: remove_object(obj)

def plane_metrics(bm, ref4, deadline=None):
    check_deadline(deadline)
    if not ref4['planos']: return []
    obj = object_from_bm(bm)
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        coordinates = np.empty(len(mesh.vertices)*3); centers = np.empty(len(mesh.polygons)*3)
        mesh.vertices.foreach_get('co', coordinates); mesh.polygons.foreach_get('center', centers)
        points = np.concatenate((coordinates.reshape(-1, 3), centers.reshape(-1, 3)))
        assignments = np.empty(len(points), dtype=int)
        for i, point in enumerate(points):
            if i % 2048 == 0: check_deadline(deadline)
            assignments[i] = ref4['cara_a_region'][ref4['bvh_crudo'].find_nearest(Vector(point))[2]]
        result = []
        for index, plane in enumerate(ref4['planos']):
            residual = np.abs(points[assignments == index]@plane['plano'][:3]+plane['plano'][3])
            result.append({'id': int(plane['id']), 'n': len(residual),
                           'p95_mm': float(np.percentile(residual, 95)) if len(residual) else None})
        return result
    finally:
        evaluated.to_mesh_clear(); remove_object(obj)
