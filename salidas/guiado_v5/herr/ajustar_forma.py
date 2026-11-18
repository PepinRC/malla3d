"""Ajuste inverso de la jaula para que la SubD límite (no la jaula) quede sobre la zona; usa ajustar.fit (ajuste por mínimos cuadrados en los dos sentidos).
Uso: python ajustar_forma.py <jaula.obj> <zona.obj> <guias.json> <salida.obj> [segundos=120] [pasos=4]"""
import os as _os; RAIZ = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # raíz del repositorio
import sys, json, shutil
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, RAIZ + '/nucleo/guiado_v5')
import numpy as np
import ajustar
def _ratios(vertices, faces):  # admite caras de 3 y 4 lados
    out = []
    for f in faces:
        P = vertices[list(f)]; L = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1); out.append(L.max() / max(L.min(), 1e-12))
    return np.array(out)
ajustar._ratios = _ratios
J, Z, G, OUT = sys.argv[1:5]; SEG = float(sys.argv[5]) if len(sys.argv) > 5 else 120; PAS = int(sys.argv[6]) if len(sys.argv) > 6 else 4
def leer(p):
    V, F = [], []
    for l in open(p, encoding='utf-8'):
        if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith('f '): F.append([int(x.split('/')[0]) - 1 for x in l.split()[1:]])
    return np.array(V), F
V, F = leer(J); ZV, ZF = leer(Z)
cr = [tuple(e) for e in json.load(open(J[:-4] + '_crease.json'))['crease']]
lineas = [np.array(g['puntos']) for g in json.load(open(G, encoding='utf-8'))]
_cf = ajustar.j.crossed_faces
_n0, _ = ajustar.j.normales(V, F); _ini = _cf(V, F, _n0).copy()
print('cruzadas al inicio', int(_ini.sum()), file=sys.stderr)
ajustar.j.crossed_faces = lambda v, f, n=None: _cf(v, f, n) & ~_ini  # solo cuentan las nuevas
r = ajustar.fit(V, F, cr, ZV, ZF, lineas, (), seconds=SEG, steps=PAS)
W, info = r if isinstance(r, tuple) else (r, {})
W = np.asarray(W)
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('# ajuste inverso\n')
    for p in W: fh.write('v %.6f %.6f %.6f\n' % tuple(p))
    for f in F: fh.write('f ' + ' '.join(str(i + 1) for i in f) + '\n')
shutil.copy(J[:-4] + '_crease.json', OUT[:-4] + '_crease.json')
print(json.dumps(info if isinstance(info, dict) else {'info': str(info)}, default=str)[:800])
