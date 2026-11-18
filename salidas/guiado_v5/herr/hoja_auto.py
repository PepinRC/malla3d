"""Hoja del auto completo: filas = vistas, columnas = Tripo / automática / semiautomática.
Uso: python hoja_auto.py <carpeta_img> <salida.png> <titulo_auto> <titulo_semi>"""
import sys
from PIL import Image, ImageDraw, ImageFont
D, OUT, TA, TS = sys.argv[1:5]
cols = [('tripo', '1. Tripo (original)'), ('automatica', TA), ('semi', TS)]
vistas = ['frente34', 'lado', 'atras34']
w, h = 960, 600; top = 46
out = Image.new('RGB', (3 * w + 20, len(vistas) * h + top), 'white'); d = ImageDraw.Draw(out)
f = ImageFont.truetype('arial.ttf', 28)
for c, (pref, t) in enumerate(cols):
    d.text((c * (w + 10) + 10, 8), t, fill='black', font=f)
    for r, v in enumerate(vistas):
        im = Image.open('%s/%s_%s.png' % (D, pref, v)).convert('RGB').crop((160, 60, 1440, 860)).resize((w, h))
        out.paste(im, (c * (w + 10), top + r * h))
out.save(OUT); print(out.size)
