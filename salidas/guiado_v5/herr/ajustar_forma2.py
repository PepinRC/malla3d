"""Como ajustar_forma.py, pero sin el freno de 'no sumar ni una cara estirada' (se mide después).
Uso: python ajustar_forma2.py <jaula.obj> <zona.obj> <guias.json> <salida.obj> [segundos] [pasos]"""
import os as _os; RAIZ = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # raíz del repositorio
import sys, runpy, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, RAIZ + '/nucleo/guiado_v5')
import ajustar
ajustar._ratios = lambda vertices, faces: np.zeros(len(faces))
sys.argv = [sys.argv[0]] + sys.argv[1:]
src = open(RAIZ + '/salidas/guiado_v5/herr/ajustar_forma.py', encoding='utf-8').read().replace('ajustar._ratios = _ratios', 'pass')
exec(compile(src, 'ajustar_forma.py', 'exec'), {'__name__': '__main__', '__file__': RAIZ + '/salidas/guiado_v5/herr/ajustar_forma.py'})
