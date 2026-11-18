"""Funciones copiadas de jaula.py para aislar el frente lowpoly.
SHA256 origen: 23975791c24104e63b5f5349651bdbf9414731d2a2b3ad4d2915458245c0d72d
"""
import numpy as np
from collections import defaultdict
from pathlib import Path
import json
def leer_obj(path):
    vertices, faces = [], []
    for lineno, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        fields = line.partition('#')[0].split()
        if not fields:
            continue
        if fields[0] == 'v':
            if len(fields) < 4:
                raise ValueError(f'{path}:{lineno}: vertice incompleto')
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == 'f':
            face = []
            for field in fields[1:]:
                idx = int(field.split('/')[0])
                if idx == 0:
                    raise ValueError(f'{path}:{lineno}: indice OBJ cero')
                face.append(idx - 1 if idx > 0 else len(vertices) + idx)
            if len(face) < 3:
                raise ValueError(f'{path}:{lineno}: cara incompleta')
            faces.append(face)
    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if not len(v) or not faces or not np.isfinite(v).all():
        raise ValueError('OBJ vacio o coordenadas no finitas')
    if min(min(f) for f in faces) < 0 or max(max(f) for f in faces) >= len(v):
        raise ValueError('Indice OBJ fuera de rango')
    return v, faces

def aristas(faces):
    edges = defaultdict(list)
    for fi, f in enumerate(faces):
        for a, b in zip(f, f[1:] + f[:1]):
            edges[tuple(sorted((a, b)))].append(fi)
    return dict(edges)

def normales(v, faces):
    n = np.array([np.cross(v[f[1:]] - v[f[0]], v[f[2:] + f[:1]] - v[f[0]]).sum(axis=0)
                  for f in faces])
    lengths = np.linalg.norm(n, axis=1)
    return n / np.maximum(lengths[:, None], 1e-30), lengths * .5

def triangular(v, face):
    """Ear clipping en el plano dominante; admite poligonos concavos simples."""
    if len(face) == 3:
        return [list(face)]
    if len(face) == 4:
        # Los cuadros SubD pueden ser no planos. Elige la diagonal con menos
        # inversion; tambien permite medir triangulos degenerados sin abortar.
        a,b,c,d = face
        choices = [([a,b,c],[a,c,d]),([a,b,d],[b,c,d])]
        def quality(pair):
            normals = [np.cross(v[t[1]]-v[t[0]],v[t[2]]-v[t[0]]) for t in pair]
            lengths = [np.linalg.norm(n) for n in normals]
            return min(lengths) * (1 + float(normals[0] @ normals[1])/max(lengths[0]*lengths[1],1e-30))
        return list(max(choices,key=quality))
    p = v[face]
    n = np.cross(p - p[0], np.roll(p, -1, axis=0) - p[0]).sum(axis=0)
    p = np.delete(p, np.argmax(np.abs(n)), axis=1)
    def cross(a, b):
        return a[0]*b[1] - a[1]*b[0]
    area = sum(cross(a, b) for a, b in zip(p, np.roll(p, -1, axis=0)))
    sign = 1 if area >= 0 else -1
    ids, result = list(range(len(face))), []
    while len(ids) > 3:
        ears = []
        for i, b in enumerate(ids):
            a, c = ids[i-1], ids[(i+1) % len(ids)]
            if sign*cross(p[b]-p[a], p[c]-p[b]) <= 1e-12:
                continue
            inside = False
            for d in ids:
                if d in (a, b, c):
                    continue
                if min(sign*cross(p[y]-p[x], p[d]-p[x]) for x, y in ((a,b),(b,c),(c,a))) >= -1e-12:
                    inside = True
                    break
            if not inside:
                lengths = [np.linalg.norm(p[x]-p[y]) for x,y in ((a,b),(b,c),(c,a))]
                ears.append((max(lengths)/max(min(lengths), 1e-20), i, [face[a], face[b], face[c]]))
        if not ears:
            raise ValueError('Contorno degenerado, no simple o no triangulable')
        _, i, tri = min(ears)
        result.append(tri)
        ids.pop(i)
    result.append([face[i] for i in ids])
    return result

def unir_contornos(a, b):
    edges = [(x,y) for f in (a,b) for x,y in zip(f, f[1:]+f[:1])]
    counts = defaultdict(int)
    for x,y in edges:
        counts[tuple(sorted((x,y)))] += 1
    edges = [(x,y) for x,y in edges if counts[tuple(sorted((x,y)))] == 1]
    nxt = dict(edges)
    if len(nxt) != len(edges) or not edges:
        return None
    start = edges[0][0]
    loop, at = [], start
    while at not in loop:
        loop.append(at)
        if at not in nxt:
            return None
        at = nxt[at]
    return loop if at == start and len(loop) == len(edges) else None

def aspecto(v, face):
    p = v[face]
    lens = np.linalg.norm(p-np.roll(p,1,axis=0), axis=1)
    return float(lens.max()/max(lens.min(),1e-30))

def subdividir(v, faces, crease, levels=3):
    """Catmull-Clark uniforme, crease infinito y regla de borde suave."""
    for _ in range(levels):
        ed = aristas(faces)
        sharp = {tuple(sorted(e)) for e in crease} | {e for e,fs in ed.items() if len(fs) == 1}
        fp = np.array([v[f].mean(axis=0) for f in faces])
        ve, vf, vc = defaultdict(list), defaultdict(list), defaultdict(list)
        for fi,f in enumerate(faces):
            for a in f:
                vf[a].append(fi)
        points = v.copy()
        ep = {}
        extra = []
        for (a,b),fs in ed.items():
            ve[a].append(b); ve[b].append(a)
            ep[(a,b)] = len(v)+len(extra)
            if (a,b) in sharp:
                extra.append((v[a]+v[b])*.5)
                vc[a].append(b); vc[b].append(a)
            else:
                extra.append((v[a]+v[b]+fp[fs].sum(axis=0))/4)
        for a,ns in ve.items():
            cs = vc[a]
            if len(cs) > 2 or len(vf[a]) == 1:
                continue
            if len(cs) == 2:
                points[a] = (6*v[a]+v[cs].sum(axis=0))/8
            else:
                k = len(ns)
                points[a] = (fp[vf[a]].mean(axis=0) + v[ns].mean(axis=0) + (k-2)*v[a])/k
        newfaces = []
        for fi,f in enumerate(faces):
            center = len(v)+len(extra)+fi
            for i,a in enumerate(f):
                newfaces.append([a,ep[tuple(sorted((a,f[(i+1)%len(f)])))],center,ep[tuple(sorted((a,f[i-1])))]] )
        crease = [(a,ep[e]) for e in sorted(sharp) for a in e]
        v = np.vstack((points,np.asarray(extra),fp))
        faces = newfaces
    return v, faces

def dist_triangulos(points, triangles, closest=False):
    """Distancia exacta punto-triangulo por proyeccion interior y segmentos."""
    p = points[:,None,:]
    a,b,c = triangles[:,0],triangles[:,1],triangles[:,2]
    ab,ac = b-a,c-a
    n = np.cross(ab,ac)
    n2 = np.sum(n*n,axis=1)
    ap = p-a
    d = np.sum(ap*n,axis=2)/np.maximum(n2,1e-30)
    projected = p-d[:,:,None]*n
    inside = np.ones(d.shape,dtype=bool)
    best = np.full(d.shape,np.inf)
    locations = np.zeros((*d.shape,3)) if closest else None
    for x,y in ((a,b),(b,c),(c,a)):
        edge = y-x
        inside &= np.sum(np.cross(edge,projected-x)*n,axis=2) >= -1e-12
        t = np.clip(np.sum((p-x)*edge,axis=2)/np.maximum(np.sum(edge*edge,axis=1),1e-30),0,1)
        dd = p-(x+t[:,:,None]*edge)
        distance = np.sum(dd*dd,axis=2)
        if closest:
            locations = np.where((distance<best)[:,:,None],p-dd,locations)
        best = np.minimum(best,distance)
    plane = inside & (n2[None,:]>1e-25) & (d*d*n2<best)
    if closest:
        locations = np.where(plane[:,:,None],projected,locations)
    best = np.where(plane,d*d*n2,best)
    return (best,locations) if closest else best

class BVH:
    """AABB de numpy: consultas por lotes, sin scipy ni ejecutables externos."""
    def __init__(self,v,faces):
        self.tri = np.array([v[t] for f in faces for t in triangular(v,f)])
        centers = self.tri.mean(axis=1)
        def build(ids):
            lo = self.tri[ids].min(axis=(0,1)); hi = self.tri[ids].max(axis=(0,1))
            if len(ids) <= 12:
                return lo,hi,ids,None,None
            axis = int(np.argmax(np.ptp(centers[ids],axis=0)))
            ids = ids[np.argsort(centers[ids,axis],kind='stable')]
            half = len(ids)//2
            return lo,hi,None,build(ids[:half]),build(ids[half:])
        self.root = build(np.arange(len(self.tri)))

    def distance(self,points,closest=False):
        best = np.full(len(points),np.inf)
        positions = np.zeros_like(points)
        def boxdist(node,ids):
            delta = np.maximum(np.maximum(node[0]-points[ids],points[ids]-node[1]),0)
            return np.sum(delta*delta,axis=1)
        def visit(node,ids):
            ids = ids[boxdist(node,ids) < best[ids]]
            if not len(ids):
                return
            lo,hi,tri,left,right = node
            if tri is not None:
                if closest:
                    ds,locations = dist_triangulos(points[ids],self.tri[tri],True)
                    k = ds.argmin(axis=1); rows=np.arange(len(ids)); values=ds[rows,k]
                    improve = values<best[ids]
                    positions[ids[improve]] = locations[rows[improve],k[improve]]
                else:
                    values = dist_triangulos(points[ids],self.tri[tri]).min(axis=1)
                best[ids] = np.minimum(best[ids],values)
                return
            nearleft = boxdist(left,ids) <= boxdist(right,ids)
            a,b = ids[nearleft],ids[~nearleft]
            if len(a):
                visit(left,a); visit(right,a)
            if len(b):
                visit(right,b); visit(left,b)
        for start in range(0,len(points),512):
            visit(self.root,np.arange(start,min(start+512,len(points))))
        return (np.sqrt(best),positions) if closest else np.sqrt(best)

def muestras(v, faces, count=6000, seed=41):
    ts = np.array([t for f in faces for t in triangular(v,f)])
    tri = v[ts]
    area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    if area.sum() <= 0:
        raise ValueError('Superficie sin area')
    rng = np.random.default_rng(seed)
    selected = tri[rng.choice(len(tri),count,p=area/area.sum())]
    u,w = rng.random((2,count))
    u = np.sqrt(u)
    return selected[:,0]*(1-u[:,None])+selected[:,1]*(u*(1-w))[:,None]+selected[:,2]*(u*w)[:,None]

def crossed_faces(v,faces,normals=None):
    """Rechaza contornos proyectados no simples; admite esquinas rectas colineales."""
    if normals is None:
        normals=normales(v,faces)[0]
    flags=[]
    for q,n in zip(faces,normals):
        p=v[q].copy()
        if np.min(np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1))<=1e-9:
            flags.append(True); continue
        if np.linalg.norm(n)<1e-12:
            candidates=np.cross(p[1:]-p[0],np.roll(p[1:],-1,axis=0)-p[0])
            n=candidates[np.argmax(np.linalg.norm(candidates,axis=1))]
            n=n/max(np.linalg.norm(n),1e-30)
        p-=np.outer((p-p[0])@n,n)
        eps=max(1.,float(np.max(np.linalg.norm(p-p[0],axis=1)))**2)*1e-12
        def turn(a,b,c):
            return float(np.cross(b-a,c-a)@n)
        def on_segment(a,b,c):
            return abs(turn(a,b,c))<=eps and float((c-a)@(c-b))<=eps
        bad=False
        for a,b,c,d in ((p[0],p[1],p[2],p[3]),(p[1],p[2],p[3],p[0])):
            bad |= (turn(a,b,c)*turn(a,b,d)<-eps*eps and turn(c,d,a)*turn(c,d,b)<-eps*eps)
            bad |= on_segment(a,b,c) or on_segment(a,b,d) or on_segment(c,d,a) or on_segment(c,d,b)
        flags.append(bad)
    return np.array(flags,dtype=bool)


def escribir(path,v,faces,crease,info,precision=8):
    if precision not in (8,12):
        raise ValueError('Precision OBJ soportada: 8 o 12 decimales')
    if not len(v) or not faces or not np.isfinite(v).all():
        raise ValueError('La jaula no contiene geometria finita')
    if any(len(f)!=4 or len(set(f))!=4 or min(f)<0 or max(f)>=len(v) for f in faces):
        raise ValueError('La jaula contiene cuadros invalidos')
    edges = aristas(faces)
    if any(len(fs)>2 for fs in edges.values()) or any(tuple(sorted(e)) not in edges for e in crease):
        raise ValueError('Topologia o indices crease invalidos')
    if np.any(normales(v,faces)[1]<1e-10):
        raise ValueError('La jaula contiene caras sin area')
    if crossed_faces(v,faces).any():
        raise ValueError('La jaula contiene contornos no simples o aristas nulas')
    serialized=np.array([[float(f'{x:.{precision}f}') for x in p] for p in v])
    if np.any(normales(serialized,faces)[1]<1e-10) or crossed_faces(serialized,faces).any():
        raise ValueError('El redondeo OBJ invalida una cara; aumentar precision')
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    body = '# Jaula de control en mm; indices OBJ base 1\n'
    body += ''.join('v '+' '.join(f'{x:.{precision}f}' for x in p)+'\n' for p in v)
    body += ''.join('f '+' '.join(str(x+1) for x in f)+'\n' for f in faces)
    path.write_text(body,encoding='utf-8')
    path.with_name(path.stem+'_crease.json').write_text(json.dumps({'crease':[list(e) for e in crease]},indent=2)+'\n',encoding='utf-8')
    path.with_name(path.stem+'_info.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
