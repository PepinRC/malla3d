"""Jaulas Smart: emparejado, división de bandas e inversión de Catmull–Clark."""
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
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('lowpoly_base', HERE/'jaula_base.py')
j = importlib.util.module_from_spec(spec)
spec.loader.exec_module(j)


def orient(faces, conflicts_only=False):
    edges = j.aristas(faces)
    if any(len(fs)>2 for fs in edges.values()):
        raise ValueError('Entrada con arista no manifold')
    adj = defaultdict(list)
    for (a,b), fs in edges.items():
        if len(fs)==2:
            i,k = fs
            same = ((a,b) in list(zip(faces[i],faces[i][1:]+faces[i][:1]))) == ((a,b) in list(zip(faces[k],faces[k][1:]+faces[k][:1])))
            adj[i].append((k,same)); adj[k].append((i,same))
    flip = {}; conflicts=[]
    for seed in range(len(faces)):
        if seed in flip:
            continue
        flip[seed]=False; todo=[seed]
        while todo:
            i=todo.pop()
            for k,same in adj[i]:
                value=flip[i]^same
                if k in flip and flip[k]!=value:
                    conflicts.append(tuple(sorted((i,k))))
                if k not in flip:
                    flip[k]=value; todo.append(k)
    if conflicts_only:
        return sorted(set(conflicts))
    if conflicts:
        raise ValueError('Entrada no orientable')
    return [f[::-1] if flip[i] else list(f) for i,f in enumerate(faces)]


def separate_sheets(v,f):
    """Separa conexiones no manifold conservando cada polígono y su posición."""
    v=v.tolist(); f=[q.copy() for q in f]; detached=[]
    while True:
        bad={e:fs for e,fs in j.aristas(f).items() if len(fs)>2}
        if not bad:
            break
        counts=defaultdict(int)
        for fs in bad.values():
            for fi in fs:
                counts[fi]+=1
        fi=max(counts,key=lambda i:(counts[i],-len(f[i]),-i))
        new=[]
        for a in f[fi]:
            new.append(len(v)); v.append(v[a].copy())
        f[fi]=new; detached.append(fi)
    while True:
        conflicts=orient(f,conflicts_only=True)
        if not conflicts:
            break
        _,areas=j.normales(np.array(v),f)
        fi=min({a for pair in conflicts for a in pair},key=lambda a:areas[a])
        new=[]
        for a in f[fi]:
            new.append(len(v)); v.append(v[a].copy())
        f[fi]=new; detached.append(fi)
    # Un vértice compartido sólo en un punto también requiere hojas distintas.
    edges=j.aristas(f); incident=defaultdict(set)
    for fi,q in enumerate(f):
        for a in q:
            incident[a].add(fi)
    for a,ids in incident.items():
        remaining=set(ids); first=True
        while remaining:
            todo=[min(remaining)]; component=set()
            while todo:
                fi=todo.pop()
                if fi not in remaining:
                    continue
                remaining.remove(fi); component.add(fi)
                for b in f[fi]:
                    if b!=a:
                        todo.extend(k for k in edges.get(tuple(sorted((a,b))),[]) if k in remaining)
            if not first:
                new=len(v); v.append(v[a].copy())
                for fi in component:
                    f[fi]=[new if x==a else x for x in f[fi]]
            first=False
    return np.array(v),f,detached


def topology(v,f):
    ed=j.aristas(f); degree=defaultdict(int)
    border={a for e,fs in ed.items() if len(fs)==1 for a in e}
    for a,b in ed:
        degree[a]+=1; degree[b]+=1
    return dict(vertices=len(v), caras=len(f), cuadros_pct=100*sum(len(x)==4 for x in f)/len(f),
                aspecto_mayor_3_5_pct=100*float(np.mean([j.aspecto(v,x)>3.5 for x in f])),
                polos_interiores_6_mas=sum(n>=6 and a not in border for a,n in degree.items()),
                aristas_no_manifold=sum(len(fs)>2 for fs in ed.values()))


def crossed_faces(v,f,normals=None):
    return j.crossed_faces(v,f,normals)


def pair_triangles(v,faces):
    edges=j.aristas(faces); candidates=[]
    n,_=j.normales(v,faces)
    for edge,fs in edges.items():
        if len(fs)!=2 or any(len(faces[i])!=3 for i in fs):
            continue
        a,b=fs; quad=j.unir_contornos(faces[a],faces[b])
        if quad is None or len(quad)!=4 or n[a]@n[b]<math.cos(math.radians(25)):
            continue
        p=v[quad]; crosses=np.cross(np.roll(p,-1,axis=0)-p,np.roll(p,-2,axis=0)-np.roll(p,-1,axis=0))
        if np.any(crosses@(n[a]+n[b])<=1e-10):
            continue
        candidates.append((j.aspecto(v,quad),a,b,quad))
    taken=set(); out=[]
    for score,a,b,q in sorted(candidates):
        if a not in taken and b not in taken and score<=5:
            taken.update((a,b)); out.append(q)
    return [f for i,f in enumerate(faces) if i not in taken]+out


def strip(faces,start):
    """Propaga una división por lados opuestos; finaliza en triángulo o borde."""
    edges=j.aristas(faces); seen=set(); queue=[tuple(sorted(start))]
    while queue:
        e=queue.pop()
        if e in seen:
            continue
        seen.add(e)
        for fi in edges[e]:
            f=faces[fi]
            if len(f)==3:
                continue
            index=next(i for i in range(4) if tuple(sorted((f[i],f[(i+1)%4])))==e)
            queue.append(tuple(sorted((f[(index+2)%4],f[(index+3)%4]))))
    return seen


def split_strip(v,faces,selected):
    mids={}; vertices=v.tolist()
    for a,b in sorted(selected):
        mids[a,b]=len(vertices); vertices.append(((v[a]+v[b])*.5).tolist())
    out=[]
    for f in faces:
        hits=[i for i in range(len(f)) if tuple(sorted((f[i],f[(i+1)%len(f)]))) in selected]
        if not hits:
            out.append(f); continue
        if len(f)==3 and len(hits)==1:
            i=hits[0]; a,b,c=f[i:]+f[:i]
            out.append([a,mids[tuple(sorted((a,b)))],b,c])
        elif len(f)==4 and len(hits)==2 and (hits[1]-hits[0])==2:
            i=hits[0]; a,b,c,d=f[i:]+f[:i]
            x=mids[tuple(sorted((a,b)))]; y=mids[tuple(sorted((c,d)))]
            out.extend([[a,x,y,d],[x,b,c,y]])
        elif len(f)==4 and len(hits)==4:
            center=len(vertices); vertices.append(v[f].mean(axis=0).tolist())
            for i,a in enumerate(f):
                out.append([a,mids[tuple(sorted((a,f[(i+1)%4])))],center,mids[tuple(sorted((a,f[i-1])))]] )
        else:
            raise ValueError('Banda incompatible: número impar de cortes')
    return np.array(vertices),out


def split_integer(v,faces,selected,n):
    vertices=v.tolist(); chains={}
    for a,b in sorted(selected):
        chains[a,b]=[a]
        for k in range(1,n):
            chains[a,b].append(len(vertices)); vertices.append(((1-k/n)*v[a]+k/n*v[b]).tolist())
        chains[a,b].append(b)
    def chain(a,b):
        return chains[a,b] if a<b else chains[b,a][::-1]
    out=[]
    for f in faces:
        hits=[i for i in range(4) if tuple(sorted((f[i],f[(i+1)%4]))) in selected]
        if not hits:
            out.append(f)
        elif len(hits)==2 and hits[1]-hits[0]==2:
            k=hits[0]; a,b,c,d=f[k:]+f[:k]; top=chain(a,b); bottom=chain(d,c)
            out.extend([[top[i],top[i+1],bottom[i+1],bottom[i]] for i in range(n)])
        elif len(hits)==4:
            a,b,c,d=f; grid={}
            for i,x in enumerate(chain(a,b)): grid[i,0]=x
            for i,x in enumerate(chain(d,c)): grid[i,n]=x
            for k,x in enumerate(chain(a,d)): grid[0,k]=x
            for k,x in enumerate(chain(b,c)): grid[n,k]=x
            for k in range(1,n):
                for i in range(1,n):
                    s,t=i/n,k/n; grid[i,k]=len(vertices)
                    vertices.append(((1-s)*(1-t)*v[a]+s*(1-t)*v[b]+s*t*v[c]+(1-s)*t*v[d]).tolist())
            out.extend([[grid[i,k],grid[i+1,k],grid[i+1,k+1],grid[i,k+1]] for k in range(n) for i in range(n)])
        else:
            raise ValueError('División entera requiere banda conforme de cuadros')
    return np.array(vertices),out


def improve_integer(v,f,budget=150):
    history=[]
    for _ in range(35):
        ratios=np.array([j.aspecto(v,q) for q in f]); bad=np.flatnonzero(ratios>3.5)
        if len(bad)<=.1*len(f) or len(f)>=budget:
            break
        old=len(bad)-.1*len(f); candidates=[]; seen=set()
        for fi in bad[np.argsort(-ratios[bad])][:40]:
            q=f[fi]; lengths=[np.linalg.norm(v[a]-v[b]) for a,b in zip(q,q[1:]+q[:1])]
            k=int(np.argmax(lengths)); cuts=strip(f,(q[k],q[(k+1)%4])); key=tuple(sorted(cuts))
            if key in seen:
                continue
            seen.add(key)
            counts=[sum(tuple(sorted(e)) in cuts for e in zip(q,q[1:]+q[:1])) for q in f]
            for n in range(2,25):
                predicted=sum(n*n if c==4 else n if c==2 else 1 for c in counts)
                if predicted>budget:
                    break
                vv,ff=split_integer(v,f,cuts,n)
                new=np.array([j.aspecto(vv,q) for q in ff]); score=np.sum(new>3.5)-.1*len(ff)
                if score<old-1e-7:
                    gain=(old-score)/(len(ff)-len(f))
                    candidates.append((gain,-float(np.max(new)),vv,ff,n))
        if not candidates:
            break
        _,_,v,f,n=max(candidates,key=lambda x:x[:2]); history.append(dict(caras=len(f),segmentos=n))
    return v,f,history


def quadrangulate(v,f):
    f=pair_triangles(v,orient(f)); history=[]
    while any(len(x)==3 for x in f):
        ti=next(i for i,x in enumerate(f) if len(x)==3)
        candidates=[]
        for a,b in zip(f[ti],f[ti][1:]+f[ti][:1]):
            selected=strip(f,(a,b))
            try:
                vv,ff=split_strip(v,f,selected)
            except ValueError:
                continue
            ratios=np.array([j.aspecto(vv,q) for q in ff])
            # Castiga sobre todo las caras extremas, sin multiplicar todo el panel.
            cost=np.sum(np.maximum(ratios-3.5,0))+.15*len(ff)
            candidates.append((cost,vv,ff,len(selected)))
        if not candidates:
            raise ValueError('No se encontró banda conforme para el triángulo')
        _,v,f,count=min(candidates,key=lambda x:x[0]); history.append(count)
    return v,f,history


def signature(f):
    ed=j.aristas(f); adj=defaultdict(set)
    for a,b in ed:
        adj[a].add(b); adj[b].add(a)
    remaining=set(adj); components=0
    while remaining:
        components+=1; todo=[next(iter(remaining))]
        while todo:
            a=todo.pop()
            if a in remaining:
                remaining.remove(a); todo.extend(adj[a]&remaining)
    return len(adj)-len(ed)+len(f),components


def directed_collapse(v,f,panel,source):
    data=json.loads((HERE/'colapsos_dirigidos.json').read_text(encoding='utf-8'))
    if panel not in data:
        return v,f,[]
    recipe=data[panel]
    if sha(source)!=recipe['sha256_entrada']:
        raise ValueError('La receta topológica no corresponde a esta entrada')
    mapping={a-1:b-1 for a,b in recipe['vertices_obj']}
    ff=[]
    for q in f:
        clean=[]
        for a in q:
            a=mapping.get(a,a)
            if not clean or a!=clean[-1]:
                clean.append(a)
        if len(clean)>1 and clean[0]==clean[-1]:
            clean.pop()
        if len(clean)>=3:
            if len(set(clean))!=len(clean):
                raise ValueError('Receta con colapso no simple')
            ff.append(clean)
    if signature(ff)!=signature(f):
        raise ValueError('Receta cambia componentes o característica de Euler')
    forward=j.BVH(v,f).distance(j.muestras(v,ff,12000,41))
    backward=j.BVH(v,ff).distance(j.muestras(v,f,12000,42))
    if max(forward.mean(),backward.mean())>.15 or max(np.percentile(forward,95),np.percentile(backward,95))>.4:
        raise ValueError('Receta excede guardia geométrica')
    used=sorted({a for q in ff for a in q}); remap={a:i for i,a in enumerate(used)}
    info=dict(receta=recipe,caras_antes=len(f),caras_despues=len(ff),
              media_directa_mm=float(forward.mean()),p95_directo_mm=float(np.percentile(forward,95)),
              media_inversa_mm=float(backward.mean()),p95_inverso_mm=float(np.percentile(backward,95)))
    return v[used],[[remap[a] for a in q] for q in ff],[info]


def simplify_bands(v,f,reference,max_steps=30):
    """Quita bandas redundantes, conserva dobleces y controla cobertura en ambos sentidos."""
    history=[]; target=j.BVH(v,f); target_points=j.muestras(v,f,2500,19)
    original_vertices=v.copy(); sig=signature(f)
    for _ in range(max_steps):
        sharp={e for e in design_crease(v,f,reference) if len(j.aristas(f)[e])==2}
        protected={a for e in sharp for a in e}
        _,areas=j.normales(v,f); weight=defaultdict(float)
        for q,area in zip(f,areas):
            for a in q:
                weight[a]+=area
        ed=j.aristas(f); candidates=[]; seen=set()
        for e in sorted(ed,key=lambda x:np.linalg.norm(v[x[0]]-v[x[1]])):
            if e in sharp or np.linalg.norm(v[e[0]]-v[e[1]])>3:
                continue
            cuts=strip(f,e); key=tuple(sorted(cuts))
            if key in seen:
                continue
            seen.add(key); mapping={}; selected=[]
            for a,b in sorted(cuts):
                if (a,b) in sharp or (a in protected and b in protected):
                    continue
                if np.linalg.norm(v[a]-v[b])>4:
                    continue
                anchor=a if a in protected else b if b in protected else max((a,b),key=lambda x:weight[x])
                removed=b if anchor==a else a
                if removed in mapping or anchor in mapping or any(x==removed for x in mapping.values()):
                    continue
                mapping[removed]=anchor; selected.append((removed,anchor))
            if not mapping:
                continue
            ff=[]; valid=True
            for q in f:
                mapped=[mapping.get(a,a) for a in q]; clean=[]
                for a in mapped:
                    if not clean or clean[-1]!=a:
                        clean.append(a)
                if len(clean)>1 and clean[0]==clean[-1]:
                    clean.pop()
                if len(set(clean))!=len(clean):
                    valid=False; break
                if len(clean)>=3:
                    ff.append(clean)
            if not valid or not ff or len(ff)>=len(f) or len({tuple(sorted(q)) for q in ff})!=len(ff):
                continue
            if signature(ff)!=sig or any(len(fs)>2 for fs in j.aristas(ff).values()):
                continue
            try:
                ff=orient(ff)
                if min(j.normales(v,ff)[1])<1e-7:
                    continue
                trial=j.BVH(v,ff)
                backward=trial.distance(np.vstack((target_points,original_vertices)))
                if np.percentile(backward,95)>.3 or backward.max()>.7:
                    continue
                forward=target.distance(j.muestras(v,ff,1800,23))
                if np.percentile(forward,95)>.3 or forward.max()>.7:
                    continue
            except ValueError:
                continue
            gain=len(f)-len(ff)
            score=gain/(.1+float(forward.mean()+backward.mean()))
            candidates.append((score,ff,selected,float(np.percentile(backward,95)),float(np.percentile(forward,95))))
        if not candidates:
            break
        _,f,selected,back,fwd=max(candidates,key=lambda x:x[0])
        history.append(dict(colapsos=[list(x) for x in selected],caras=len(f),p95_ref_a_malla_mm=back,p95_malla_a_ref_mm=fwd))
    used=sorted({a for q in f for a in q}); remap={a:i for i,a in enumerate(used)}
    return v[used],[[remap[a] for a in q] for q in f],history


def improve_aspect(v,f,budget=150):
    history=[]
    for iteration in range(30):
        ratios=np.array([j.aspecto(v,q) for q in f]); bad=np.flatnonzero(ratios>3.5)
        if len(bad)<=.1*len(f) or len(f)>=budget:
            break
        old=np.sum(np.maximum(ratios-3.5,0)); candidates=[]; seen=set()
        for fi in bad[np.argsort(-ratios[bad])][:18]:
            q=f[fi]
            lengths=[np.linalg.norm(v[a]-v[b]) for a,b in zip(q,q[1:]+q[:1])]
            i=int(np.argmax(lengths)); cuts=strip(f,(q[i],q[(i+1)%4])); key=tuple(sorted(cuts))
            if key in seen:
                continue
            seen.add(key)
            vv,ff=split_strip(v,f,cuts)
            if len(ff)>budget:
                continue
            new=np.array([j.aspecto(vv,q) for q in ff]); score=np.sum(np.maximum(new-3.5,0))
            if score<old-.01:
                candidates.append(((old-score)/(len(ff)-len(f)),vv,ff))
        if not candidates:
            break
        _,v,f=max(candidates,key=lambda x:x[0]); history.append(len(f))
    return v,f,history


def support_planes(v,f,reference,budget):
    """Acerca controles a una unión suave de planos sin declararla arista viva."""
    history=[]
    for _ in range(16):
        sharp=set(design_crease(v,f,reference)); ed=j.aristas(f); n,_=j.normales(v,f)
        candidates=[]
        for edge,ids in ed.items():
            if len(ids)!=2 or edge in sharp:
                continue
            angle=math.degrees(math.acos(np.clip(n[ids[0]]@n[ids[1]],-1,1)))
            if not 10<angle<25:
                continue
            for fi in ids:
                q=f[fi]
                i=next(i for i in range(4) if tuple(sorted((q[i],q[(i+1)%4])))==edge)
                cross=(q[(i+1)%4],q[(i+2)%4]); width=np.linalg.norm(v[cross[0]]-v[cross[1]])
                if width<=7.5:
                    continue
                cuts=strip(f,cross); vv,ff=split_strip(v,f,cuts)
                if len(ff)<=budget:
                    candidates.append((width,vv,ff))
        if not candidates:
            break
        _,v,f=max(candidates,key=lambda x:x[0]); history.append(len(f))
    return v,f,history


def resolve_poles(v,f,reference):
    """Sustituye estrellas de seis cuadros por un parche 2x4, sin cortar la hoja."""
    history=[]
    for _ in range(12):
        ed=j.aristas(f); border={a for e,fs in ed.items() if len(fs)==1 for a in e}
        degree=defaultdict(int)
        for a,b in ed:
            degree[a]+=1; degree[b]+=1
        poles=[a for a,n in degree.items() if n>=6 and a not in border]
        if not poles:
            break
        sharp=design_crease(v,f,reference); candidates=[]
        for a in poles:
            ids=[i for i,q in enumerate(f) if a in q]
            if degree[a]!=6 or len(ids)!=6:
                continue
            star=[f[i] for i in ids]; se=j.aristas(star); directed={}
            for q in star:
                for x,y in zip(q,q[1:]+q[:1]):
                    if len(se[tuple(sorted((x,y)))])==1:
                        directed[x]=y
            if len(directed)!=12:
                continue
            ring=[min(directed)]
            while len(ring)<12 and ring[-1] in directed:
                ring.append(directed[ring[-1]])
            if len(set(ring))!=12 or directed.get(ring[-1])!=ring[0]:
                continue
            endpoints={e[1] if e[0]==a else e[0] for e in sharp if a in e}
            if len(endpoints)>2:
                continue
            for shift in range(12):
                r=ring[shift:]+ring[:shift]
                if endpoints and not endpoints <= {r[1],r[7]}:
                    continue
                vv=v.tolist(); grid={}
                boundary=[(0,0),(1,0),(2,0),(2,1),(2,2),(2,3),(2,4),(1,4),(0,4),(0,3),(0,2),(0,1)]
                grid.update(zip(boundary,r)); grid[1,2]=a
                for y in (1,3):
                    t=y/4
                    p=.5*(v[grid[0,y]]+v[grid[2,y]])+(1-t)*v[r[1]]+t*v[r[7]]-((1-t)*(v[r[0]]+v[r[2]])+t*(v[r[8]]+v[r[6]]))*.5
                    if y==1 and r[1] in endpoints:
                        p=(v[r[1]]+v[a])*.5
                    if y==3 and r[7] in endpoints:
                        p=(v[r[7]]+v[a])*.5
                    grid[1,y]=len(vv); vv.append(p.tolist())
                patch=[[grid[x,y],grid[x+1,y],grid[x+1,y+1],grid[x,y+1]] for y in range(4) for x in range(2)]
                ff=[q for i,q in enumerate(f) if i not in ids]+patch; vv=np.array(vv)
                m=topology(vv,ff)
                if m['polos_interiores_6_mas']>=len(poles):
                    continue
                pn,pa=j.normales(vv,patch); sn,sa=j.normales(v,star)
                if np.min(pa)<1e-7 or np.any(pn@(sn*sa[:,None]).sum(axis=0)<0):
                    continue
                ratios=[j.aspecto(vv,q) for q in patch]
                candidates.append((sum(max(x-3.5,0) for x in ratios),vv,ff,a))
        if not candidates:
            break
        _,v,f,a=min(candidates,key=lambda x:x[0]); history.append(a)
    return v,f,history


def rotate_poles(v,f,reference):
    """Rota diagonales de dos cuadros: reparte una valencia alta sin crear costuras."""
    history=[]
    for _ in range(20):
        ed=j.aristas(f); sharp=set(design_crease(v,f,reference))
        border={a for e,fs in ed.items() if len(fs)==1 for a in e}
        degree=defaultdict(int)
        for a,b in ed:
            degree[a]+=1; degree[b]+=1
        poles={a for a,n in degree.items() if n>=6 and a not in border}
        if not poles:
            break
        candidates=[]
        normals,areas=j.normales(v,f)
        for edge,fs in ed.items():
            if len(fs)!=2 or edge in sharp or not poles.intersection(edge):
                continue
            a,b=fs; ring=j.unir_contornos(f[a],f[b])
            if ring is None or len(ring)!=6 or len(set(ring))!=6:
                continue
            for k in range(3):
                r=ring[k:]+ring[:k]; diagonal=tuple(sorted((r[0],r[3])))
                if diagonal in ed or any(degree[x]>=5 and x not in border for x in diagonal):
                    continue
                patch=[r[:4],r[3:]+r[:1]]
                pn,pa=j.normales(v,patch); normal=normals[fs].sum(axis=0)
                if np.min(pa)<1e-6 or np.any(pn@normal<=0):
                    continue
                # No aceptar cuadros cóncavos: una rotación no debe esconder un pliegue.
                convex=True
                for q,nn in zip(patch,pn):
                    p=v[q]; turns=np.cross(np.roll(p,-1,axis=0)-p,np.roll(p,-2,axis=0)-np.roll(p,-1,axis=0))
                    if np.any(turns@nn < -1e-8):
                        convex=False
                if not convex:
                    continue
                local=j.BVH(v,[f[a],f[b]])
                samples=j.muestras(v,patch,100,17)
                error=float(local.distance(samples).max())
                if error>1.5:
                    continue
                ff=[q for i,q in enumerate(f) if i not in fs]+patch
                m=topology(v,ff)
                if m['polos_interiores_6_mas']>=len(poles):
                    continue
                score=sum(j.aspecto(v,q) for q in patch)+error*10
                candidates.append((score,ff,edge,diagonal,error))
        if not candidates:
            break
        _,f,edge,diagonal,error=min(candidates,key=lambda x:x[0])
        history.append(dict(retirada=list(edge),nueva=list(diagonal),error_local_max_mm=error))
    return v,f,history


def design_crease(v,f,reference=None,angle=45):
    ed=j.aristas(f); normals,_=j.normales(v,f)
    if reference:
        data=json.loads(Path(reference).read_text(encoding='utf-8-sig'))
        segments=np.array([[a,b] for line in data['lineas_doblez'] for a,b in zip(line['puntos'][:-1],line['puntos'][1:])]).reshape(-1,2,3)
    else:
        segments=np.zeros((0,2,3))
    sharp=[]
    for e,fs in ed.items():
        if len(fs)==1:
            sharp.append(e); continue
        if reference:
            if not len(segments):
                continue
            pts=np.array([v[e[0]],v[list(e)].mean(axis=0),v[e[1]]]); a=segments[:,0]; d=segments[:,1]-a
            t=np.clip(np.sum((pts[:,None]-a)*d,axis=2)/np.maximum(np.sum(d*d,axis=1),1e-20),0,1)
            dist=np.linalg.norm(pts[:,None]-a-t[:,:,None]*d,axis=2).min(axis=1)
            if dist.max()<.15:
                sharp.append(e)
        elif normals[fs[0]]@normals[fs[1]]<math.cos(math.radians(angle)):
            sharp.append(e)
    return sharp


def planarize(v,f,reference,rv,rf,max_move=1.5):
    data=json.loads(Path(reference).read_text(encoding='utf-8-sig'))
    regions=data['regiones_planas']; labels={fi:i for i,r in enumerate(regions) for fi in r['caras']}
    triangles=[]; source_ids=[]
    for fi,q in enumerate(rf):
        for t in j.triangular(rv,q):
            triangles.append(rv[t]); source_ids.append(fi)
    centers=np.array([v[q].mean(axis=0) for q in f])
    distances=j.dist_triangulos(centers,np.array(triangles))
    nearest=np.array(source_ids)[distances.argmin(axis=1)]
    planes=[np.array(r['plano']) for r in regions]; incidence=defaultdict(set)
    for q,fi in zip(f,nearest):
        if fi in labels:
            for a in q:
                incidence[a].add(labels[fi])
    # Los planos ajustados a una entrada alabeada no siempre se cortan cerca del
    # vértice compartido: resolver también sus offsets evita intersecciones lejanas.
    rows=[]; residual=[]
    for a,ids in incidence.items():
        for i in sorted(ids):
            row=np.zeros(v.size+len(planes)); row[3*a:3*a+3]=planes[i][:3]; row[v.size+i]=1
            rows.append(row); residual.append(v[a]@planes[i][:3]+planes[i][3])
    delta=np.linalg.lstsq(np.array(rows),-np.array(residual),rcond=1e-10)[0]
    out=v+delta[:v.size].reshape(v.shape)
    offsets=delta[v.size:]
    if np.max(np.abs(offsets))>.19 or np.linalg.norm(out-v,axis=1).max()>max_move:
        return v,dict(aceptado=False,motivo='aplanado desplaza planos o controles fuera de tolerancia',offsets_mm=offsets.tolist(),movimiento_max_mm=float(np.linalg.norm(out-v,axis=1).max()))
    oldn,olda=j.normales(v,f); newn,newa=j.normales(out,f)
    if np.any(np.sum(oldn*newn,axis=1)<.5) or np.any(newa<olda*.2):
        return v,dict(aceptado=False,motivo='inversión o colapso al aplanar')
    return out,dict(aceptado=True,vertices_con_planos=len(incidence),offsets_mm=offsets.tolist(),
                    movimiento_max_mm=float(np.linalg.norm(out-v,axis=1).max()))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(panel,budget=150,uniform=0,variant=None,support=False,directed=False,planar=False,integer=False,no_collapse=False):
    start=time.perf_counter(); source=ROOT/'datos/lowpoly/a8bc0f28'/f'{panel}.obj'
    rv,rf=j.leer_obj(source); v,f,detached=separate_sheets(rv,rf)
    reference=ROOT/'evaluacion/referencia_lowpoly'/f'{panel}.json'
    if no_collapse:
        f=orient(f); simplifications=[]
    elif directed:
        v,f,simplifications=directed_collapse(v,orient(f),panel,source)
    else:
        v,f,simplifications=simplify_bands(v,orient(f),reference if reference.exists() else None)
    v,f,history=quadrangulate(v,f)
    v,f,rotations=rotate_poles(v,f,reference if reference.exists() else None)
    v,f,poles=resolve_poles(v,f,reference if reference.exists() else None)
    v,f,refinements=(improve_integer if integer else improve_aspect)(v,f,budget)
    for _ in range(uniform):
        v,f=split_strip(v,f,set(j.aristas(f)))
    supports=[]
    if support:
        v,f,supports=support_planes(v,f,reference if reference.exists() else None,budget)
    vertices_before_repair=len(v)
    v,f,late_detached=separate_sheets(v,f)
    f=orient(f)
    crease=design_crease(v,f,reference if reference.exists() else None)
    flatten=None
    if planar and reference.exists():
        v,flatten=planarize(v,f,reference,rv,rf)
    info=dict(entrada=str(source.relative_to(ROOT)),sha256_entrada=sha(source),sha256_codigo=sha(__file__),
              metodo='Emparejado de triángulos y división conforme de bandas por lados opuestos',
              referencia_forma='Malla Smart a8bc0f28; no HD',referencia_diseno=str(reference.relative_to(ROOT)) if reference.exists() else None,
              caras_separadas_no_manifold=detached,simplificacion_bandas=simplifications,bandas_triangulos=history,rotaciones_polos=rotations,polos_resueltos=poles,soportes_suaves_caras=supports,refinamientos_caras=refinements,subdivisiones_lineales=uniform,aplanado=flatten,metricas=topology(v,f),
              reparacion_final=dict(vertices_duplicados=len(v)-vertices_before_repair,caras_separadas=late_detached),
              parametros=dict(presupuesto=budget,uniforme=uniform,soportes=support,colapsos_dirigidos=directed,planarizar=planar,refinar_entero=integer,sin_colapsos=no_collapse),
              advertencias=['C5-C9 requieren medición independiente; no se certifica aceptación por estadísticas topológicas'],
              segundos=time.perf_counter()-start)
    out=ROOT/'salidas/jaula_lowpoly'/f'{panel}.obj'
    if variant:
        if not variant.replace('_','').isalnum():
            raise ValueError('Nombre de variante inválido')
        out=out.parent/'variantes'/variant/out.name
    j.escribir(out,v,f,crease,info)
    print(json.dumps(dict(panel=panel,**info['metricas'],segundos=info['segundos'])),flush=True)


def self_test():
    v=np.array([[0.,0,0],[2,0,0],[2,1,0],[0,1,0],[3,0,0]])
    f=[[0,1,2,3],[1,4,2]]
    vv,ff,h=quadrangulate(v,f)
    assert all(len(x)==4 for x in ff)
    assert max(map(len,j.aristas(ff).values()))==2
    assert sum(j.normales(vv,ff)[1])==2.5
    assert not topology(vv,ff)['polos_interiores_6_mas']
    assert np.allclose(j.subdividir(np.eye(len(vv)),ff,[],2)[0].sum(axis=1),1)
    vv,ff,det=separate_sheets(v,[[0,1,2],[1,0,3],[0,1,4]])
    assert det and max(map(len,j.aristas(ff).values()))<=2
    assert all(np.allclose(v[q],vv[r]) for q,r in zip([[0,1,2],[1,0,3],[0,1,4]],ff))
    print('SELFTEST PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--paneles',nargs='+',default=['p68','p37','p41','p30'])
    parser.add_argument('--presupuesto',type=int,default=150)
    parser.add_argument('--uniforme',type=int,default=0)
    parser.add_argument('--variante')
    parser.add_argument('--soportes',action='store_true')
    parser.add_argument('--colapsos-dirigidos',action='store_true')
    parser.add_argument('--planarizar',action='store_true')
    parser.add_argument('--refinar-entero',action='store_true')
    parser.add_argument('--sin-colapsos',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        for panel in args.paneles:
            generate(panel,args.presupuesto,args.uniforme,args.variante,args.soportes,args.colapsos_dirigidos,args.planarizar,args.refinar_entero,args.sin_colapsos)
