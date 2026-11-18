"""Banco Blender para una jaula guiada local.

Uso: blender -b --factory-startup -P nucleo/guiado_v5/medir.py --
 --jaula salida.obj --zona zona_original.obj --guias guias.json --referencia s054.json
 --salida salidas/guiado_v5/mediciones/medicion.json
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Se reutilizan las primitivas del banco; no se modifican sus evaluadores.
banco = load("guiado_v5_banco", ROOT / "evaluacion" / "evaluar_blender.py")
soporte = load("guiado_v5_soporte", ROOT / "nucleo" / "carroceria_v5" / "medir_soporte.py")


def read_obj(path):
    vertices, faces = banco.leer_obj(str(path))
    vertices = np.asarray(vertices, dtype=float)
    if not len(vertices) or not faces or not np.isfinite(vertices).all():
        raise ValueError(f"OBJ vacio o invalido: {path}")
    if any(len(face) < 3 or len(set(face)) != len(face) for face in faces):
        raise ValueError(f"Caras invalidas: {path}")
    return vertices, [list(map(int, face)) for face in faces]


def tree(vertices, faces):
    return BVHTree.FromPolygons(vertices.tolist(), [tuple(face) for face in faces], all_triangles=False)


def nearest_dist(surface, points):
    return np.asarray([surface.find_nearest(Vector(point))[3] for point in points], dtype=float)


def sample_polyline(points, step=.25):
    points = np.asarray(points, dtype=float)
    out, weights = [], []
    for a, b in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(b-a))
        if length <= 1e-10:
            continue
        count = max(1, int(np.ceil(length / step)))
        for i in range(count):
            out.append(a + (i + .5) / count * (b-a)); weights.append(length / count)
    return np.asarray(out), np.asarray(weights)


def clip_guides(guides, zone_tree, smart_tree, zone_smart_faces, epsilon=.001):
    """Conserva solamente muestras que realmente pertenecen a la zona original.

    El dominio se decide antes de ver la jaula: una guia que sale de la zona no
    puede recuperar cobertura por una arista de la jaula situada fuera de ella.
    """
    points, weights, ids, discarded = [], [], [], {}
    zone_smart_faces = set(zone_smart_faces)
    for guide in guides:
        ident = guide.get("n")
        if ident is None:
            raise ValueError("Cada guia requiere n")
        sampled, sampled_weights = sample_polyline(guide.get("puntos", []))
        if not len(sampled):
            discarded[str(ident)] = {"muestras": 0, "largo_mm": 0.0}
            continue
        zone_hits = [zone_tree.find_nearest(Vector(point)) for point in sampled]
        zone_distance = np.asarray([hit[3] for hit in zone_hits])
        smart_hits = [smart_tree.find_nearest(Vector(point)) for point in sampled]
        smart_distance = np.asarray([hit[3] for hit in smart_hits])
        smart_faces = np.asarray([hit[2] for hit in smart_hits], dtype=int)
        # W2 puede quedar 0.1--0.5 mm separada de la Smart. El punto original
        # queda si su cara Smart mas cercana es de zona; la jaula no decide el dominio.
        # Sobre una arista compartida, BVH puede devolver cualquiera de sus caras.
        # La misma proyeccion geometrica pertenece a zona aunque el ID elegido no.
        same_projection = np.linalg.norm(np.asarray([tuple(hit[0]) for hit in zone_hits])
                                         - np.asarray([tuple(hit[0]) for hit in smart_hits]), axis=1) <= 1e-5
        keep = ((sampled[:, 0] >= -epsilon) & (zone_distance <= smart_distance + epsilon)
                & (np.isin(smart_faces, list(zone_smart_faces)) | same_projection))
        points.extend(sampled[keep]); weights.extend(sampled_weights[keep]); ids.extend([ident] * int(keep.sum()))
        discarded[str(ident)] = {"muestras": int((~keep).sum()), "largo_mm": round(float(sampled_weights[~keep].sum()), 4)}
    return np.asarray(points), np.asarray(weights), ids, discarded


def read_crease(path, faces, count):
    crease_path = Path(path).with_name(Path(path).stem + "_crease.json")
    if not crease_path.is_file():
        raise ValueError(f"Falta crease estandar: {crease_path}")
    pairs = json.loads(crease_path.read_text(encoding="utf-8-sig")).get("crease")
    edges = {tuple(sorted((face[i], face[(i+1) % len(face)]))) for face in faces for i in range(len(face))}
    if not isinstance(pairs, list):
        raise ValueError("crease debe contener una lista")
    result = set()
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(i, int) or not 0 <= i < count for i in pair):
            raise ValueError("Indice crease invalido")
        edge = tuple(sorted(pair))
        if edge not in edges:
            raise ValueError("Arista crease inexistente")
        result.add(edge)
    return result


def evaluated(vertices, faces, crease):
    # La medicion tambien debe informar candidatos con triangulos/ngonos; el
    # banco puede evaluarlos aunque calidad los marque como fallo.
    bm, _ = banco.construir_bmesh_jaula(np.asarray(vertices, dtype=np.float32).astype(float).tolist(), faces, list(crease), True, 30.)
    try:
        obj = soporte.object_from_bm(bm)
    finally:
        bm.free()
    evaluated_obj = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated_obj.to_mesh()
    try:
        coords = np.empty(len(mesh.vertices)*3); centers = np.empty(len(mesh.polygons)*3)
        mesh.vertices.foreach_get("co", coords); mesh.polygons.foreach_get("center", centers)
        return coords.reshape(-1, 3), centers.reshape(-1, 3), [tuple(poly.vertices) for poly in mesh.polygons]
    finally:
        evaluated_obj.to_mesh_clear(); soporte.remove_object(obj)


def metrics(vertices, faces):
    edges = Counter(tuple(sorted((face[i], face[(i+1) % len(face)]))) for face in faces for i in range(len(face)))
    valence = Counter(vertex for edge in edges for vertex in edge)
    border = {vertex for edge, count in edges.items() if count == 1 for vertex in edge}
    colors = Counter(); ratios = []
    for face in faces:
        pts = vertices[face]; lengths = np.linalg.norm(pts-np.roll(pts, -1, axis=0), axis=1)
        ratio = float(lengths.max()/max(lengths.min(), 1e-9)); ratios.append(ratio)
        colors["verde" if ratio <= 2 and all(valence[v] == 4 or v in border for v in face) else "amarillo" if ratio <= 3.5 else "rojo"] += 1
    adjacency = {i: set() for i in range(len(faces))}
    owners = {}
    for i, face in enumerate(faces):
        for j in range(len(face)):
            owners.setdefault(tuple(sorted((face[j], face[(j+1) % len(face)]))), []).append(i)
    for values in owners.values():
        for item in values: adjacency[item].update(values)
    unseen, components = set(range(len(faces))), 0
    while unseen:
        components += 1; stack = [unseen.pop()]
        while stack:
            stack.extend(neighbor for neighbor in adjacency[stack.pop()] if neighbor in unseen and not unseen.remove(neighbor))
    return {"caras": len(faces), "cuadros": sum(len(f) == 4 for f in faces), "triangulos": sum(len(f) == 3 for f in faces),
            "ngonos": sum(len(f) > 4 for f in faces), "calidad": dict(colors), "ratio_max": round(max(ratios, default=0), 4),
            "polos_6_mas": sorted(v for v, n in valence.items() if v not in border and n >= 6),
            "polos_3_5": sorted(v for v, n in valence.items() if v not in border and n in (3, 5)),
            "aristas_borde": sum(n == 1 for n in edges.values()), "aristas_no_manifold": sum(n > 2 for n in edges.values()), "componentes": components}


def face_samples(vertices, faces, ids):
    points, owners = [], []
    for ident in ids:
        corners = vertices[faces[ident]]
        # Muestras interiores: no basarse en las cuatro esquinas evita perder un
        # plano que cruza el centro de una cara Smart grande.
        if len(corners) == 4:
            samples = [(1-u)*(1-v)*corners[0]+u*(1-v)*corners[1]+u*v*corners[2]+(1-u)*v*corners[3]
                       for u in (.1, .3, .5, .7, .9) for v in (.1, .3, .5, .7, .9)]
        elif len(corners) == 3:
            # Diez puntos interiores: el centro solo puede ocultar huecos en
            # triangulos Smart al medir la direccion zona -> SubD.
            samples = [(i*corners[0] + j*corners[1] + k*corners[2]) / 6
                       for i in range(1, 5) for j in range(1, 6-i) for k in [6-i-j]]
        else:
            samples = [np.mean(corners, axis=0)]
        points.extend(samples); owners.extend([ident]*len(samples))
    return np.asarray(points), np.asarray(owners, dtype=int)


def map_reference_faces(ref_v, ref_f, zone_v, zone_f, zone_tree):
    def face_key(vertices, face):
        return tuple(sorted(tuple(round(float(value), 5) for value in vertices[index]) for index in face))
    lookup = {face_key(ref_v, face): index for index, face in enumerate(ref_f)}
    exact = [lookup.get(face_key(zone_v, face)) for face in zone_f]
    if all(index is not None for index in exact):
        return sorted(set(exact)), "coincidencia de caras OBJ por coordenadas"
    points, owners = face_samples(ref_v, ref_f, range(len(ref_f)))
    mapped = set(owners[nearest_dist(zone_tree, points) <= .001].tolist())
    source_tree = tree(ref_v, ref_f)
    zone_points, _ = face_samples(zone_v, zone_f, range(len(zone_f)))
    for point in zone_points:
        _, _, owner, distance = source_tree.find_nearest(Vector(point))
        if owner is not None and distance <= .001:
            mapped.add(int(owner))
    return sorted(mapped), "interseccion por muestras interiores bidireccionales, tolerancia 0.001 mm (fallback)"


def c5(zone_v, zone_f, sub_v, sub_f):
    zone_tree, sub_tree = tree(zone_v, zone_f), tree(sub_v, sub_f)
    zone_points, _ = face_samples(zone_v, zone_f, range(len(zone_f)))
    # El banco usa vertices de referencia en el sentido inverso. El sondeo
    # interior adicional debe caer sobre la superficie BVH, no un bilineal
    # distinto de la triangulacion de los cuadros Smart no planos.
    zone_points = np.asarray([zone_tree.find_nearest(Vector(point))[0] for point in zone_points])
    sub_points = np.concatenate((sub_v, np.asarray([np.mean(sub_v[list(face)], axis=0) for face in sub_f])))
    forward, reverse = nearest_dist(zone_tree, sub_points), nearest_dist(sub_tree, zone_v)
    dense_reverse = nearest_dist(sub_tree, zone_points)
    def stat(values): return {"media_mm": round(float(values.mean()), 4), "p95_mm": round(float(np.percentile(values, 95)), 4), "max_mm": round(float(values.max()), 4)}
    a, b = stat(forward), stat(reverse)
    passed = all(float(values.mean()) <= .3 and float(np.percentile(values, 95)) <= 1.5
                 for values in (forward, reverse))
    return {"jaula_subd_a_zona": a, "zona_a_jaula_subd": b, "pasa": passed,
            "muestreo": "banco: vertices y centros SubD; vertices de referencia en sentido inverso",
            "sondeo_interior_zona_a_subd": stat(dense_reverse)}


def explicit_crease_object(vertices, faces, crease):
    """SubD con solo los crease declarados; borde no implica crease C8."""
    bm = bmesh.new()
    try:
        for point in vertices: bm.verts.new(point)
        bm.verts.ensure_lookup_table()
        for face in faces:
            bm.faces.new([bm.verts[index] for index in face])
        bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table(); bm.verts.index_update()
        layer = bm.edges.layers.float.new("crease_edge")
        edges = {tuple(sorted(vertex.index for vertex in edge.verts)): edge for edge in bm.edges}
        for pair in crease: edges[tuple(sorted(pair))][layer] = 1.
        bm.normal_update()
        return soporte.object_from_bm(bm)
    finally:
        bm.free()


def explicit_crease_segments(obj):
    evaluated_obj = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated_obj.to_mesh()
    try:
        if "crease_edge" not in mesh.attributes: return None
        values = np.asarray([item.value for item in mesh.attributes["crease_edge"].data])
        selected = values > .5
        if not np.any(selected): return None
        coords = np.empty(len(mesh.vertices)*3); edges = np.empty(len(mesh.edges)*2, dtype=int)
        mesh.vertices.foreach_get("co", coords); mesh.edges.foreach_get("vertices", edges)
        coords, edges = coords.reshape(-1, 3), edges.reshape(-1, 2)
        return soporte.Segments(coords[edges[selected, 0]], coords[edges[selected, 1]])
    finally:
        evaluated_obj.to_mesh_clear()


def c8(vertices, faces, crease, points, weights, ids):
    if not len(points): return {"estado": "N/D", "pasa": None, "nota": "ninguna muestra de guia cae en la zona original"}
    obj = explicit_crease_object(vertices, faces, crease)
    try:
        segments = explicit_crease_segments(obj)
        if segments is None:
            return {"estado": "falla", "pasa": False, "nota": "no hay aristas crease declaradas en JSON", "p95_mm": None,
                    "media_mm": None, "max_mm": None, "cobertura_05mm_pct": 0., "muestras": int(len(points)),
                    "largo_mm": round(float(weights.sum()), 4), "guias": sorted(set(ids)), "objetivo_crease": "solo pares crease JSON explicitos"}
        distances, _ = segments.closest(points)
    finally: soporte.remove_object(obj)
    p95 = float(banco.percentil_ponderado(distances, weights, 95)); coverage = float(100*weights[distances <= .5].sum()/weights.sum())
    return {"estado": "pasa" if p95 <= .5 and coverage >= 90 else "falla", "pasa": bool(p95 <= .5 and coverage >= 90),
            "p95_mm": round(p95, 4), "media_mm": round(float(np.average(distances, weights=weights)), 4), "max_mm": round(float(distances.max()), 4),
            "cobertura_05mm_pct": round(coverage, 4), "muestras": int(len(points)), "largo_mm": round(float(weights.sum()), 4), "guias": sorted(set(ids)),
            "objetivo_crease": "solo pares crease JSON explicitos; un borde cuenta solo si esta declarado"}


def c9(reference, ref_v, ref_f, mapped_ids, sub_points):
    ref_tree = tree(ref_v, ref_f); assigned = np.asarray([ref_tree.find_nearest(Vector(p))[2] for p in sub_points])
    planes = [p for p in reference.get("regiones_planas", []) if set(p.get("caras", [])) & set(mapped_ids)]
    result = []
    for plane in planes:
        coeff = np.asarray(plane["plano"], dtype=float); residual = np.abs(sub_points[np.isin(assigned, plane["caras"])] @ coeff[:3] + coeff[3])
        value = float(np.percentile(residual, 95)) if len(residual) else None
        result.append({"id": plane["id"], "caras_referencia": plane["caras"], "muestras": int(len(residual)), "p95_mm": round(value, 4) if value is not None else None, "estado": "pasa" if len(residual) >= 4 and value <= .2 else "falla"})
    passed = bool(result) and all(item["estado"] == "pasa" for item in result)
    return {"estado": "pasa" if passed else "falla" if result else "N/D", "pasa": passed if result else None, "por_region": result}


def measure(args):
    jaula_v, jaula_f = read_obj(args.jaula); zone_v, zone_f = read_obj(args.zona)
    crease = read_crease(args.jaula, jaula_f, len(jaula_v)); reference = json.loads(Path(args.referencia).read_text(encoding="utf-8-sig"))
    source = Path(reference["fuente"]); source = source if source.is_absolute() else ROOT / source
    expected_source = reference.get("sha256_hd")
    if not isinstance(expected_source, str) or hashlib.sha256(source.read_bytes()).hexdigest().lower() != expected_source.lower():
        raise ValueError("La fuente Smart no coincide con el SHA congelado de la referencia")
    ref_v, ref_f = read_obj(source)
    guides = json.loads(Path(args.guias).read_text(encoding="utf-8-sig"))
    if not isinstance(guides, list): raise ValueError("--guias debe ser lista")
    zone_tree = tree(zone_v, zone_f)
    mapped, mapping_method = map_reference_faces(ref_v, ref_f, zone_v, zone_f, zone_tree)
    points, weights, ids, discarded = clip_guides(guides, zone_tree, tree(ref_v, ref_f), mapped)
    sub_v, sub_c, sub_f = evaluated(jaula_v, jaula_f, crease); sub_points = np.concatenate((sub_v, sub_c))
    return {"contrato": "medir_guiado_v5", "entradas": {"jaula": str(Path(args.jaula).resolve()), "zona": str(Path(args.zona).resolve()), "guias": str(Path(args.guias).resolve()), "referencia": str(Path(args.referencia).resolve()), "referencia_fuente": str(source.resolve())},
            "sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in (("jaula", args.jaula), ("crease", Path(args.jaula).with_name(Path(args.jaula).stem + "_crease.json")), ("zona", args.zona), ("guias", args.guias), ("referencia", args.referencia))},
            "calidad": metrics(jaula_v, jaula_f), "c5": c5(zone_v, zone_f, sub_v, sub_f),
            "c8": {"recorte": {"paso_mm": .25, "dominio": "cara Smart cercana en zona o misma proyeccion en borde compartido (1e-5 mm); dist(zona)<=dist(Smart)+0.001; X>=0", "descartado_fuera_zona": discarded}, "resultado": c8(jaula_v, jaula_f, crease, points, weights, ids)},
            "c9": {"mapeo": {"metodo": mapping_method, "caras_smart": mapped}, "resultado": c9(reference, ref_v, ref_f, mapped, sub_points)}}


def validate_output(args):
    output = Path(args.salida).resolve()
    allowed = (ROOT / "salidas" / "guiado_v5" / "mediciones").resolve()
    if output == allowed or allowed not in output.parents:
        raise ValueError("--salida debe quedar bajo salidas/guiado_v5/mediciones")
    cage = Path(args.jaula).resolve()
    protected = {Path(path).resolve() for path in (args.jaula, args.zona, args.guias, args.referencia)}
    protected.update(cage.with_name(cage.stem+suffix) for suffix in ('_crease.json', '_info.json', '_grafo.json'))
    if output in protected:
        raise ValueError("La medicion no puede reemplazar una entrada ni un archivo de la jaula")
    return output


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--jaula", required=True); parser.add_argument("--zona", required=True); parser.add_argument("--guias", required=True); parser.add_argument("--referencia", required=True); parser.add_argument("--salida", required=True)
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv); output = validate_output(args); result = measure(args)
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Medicion escrita: {output}")


if __name__ == "__main__": main()
