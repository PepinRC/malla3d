"""Triangle surface projection and injective disk parameterization (NumPy/SciPy)."""
from collections import Counter, defaultdict
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def clip_positive_x(vertices, faces, seam_tolerance=0.5):
    """Clip polygons to X>=0, welding intersections; snap the near-axis boundary."""
    vertices = np.asarray(vertices, float)
    counts = Counter(tuple(sorted((a, b))) for f in faces for a, b in zip(f, list(f[1:])+[f[0]]))
    boundary = {a for e, n in counts.items() if n == 1 for a in e}
    vertices = vertices.copy()
    snap = [a for a in boundary if abs(vertices[a, 0]) <= seam_tolerance]
    snap_max=float(np.max(np.abs(vertices[snap,0]))) if snap else 0.
    vertices[snap, 0] = 0.
    out, polygons, lookup = [], [], {}
    def vertex(p):
        key = tuple(np.round(p, 10))
        if key not in lookup:
            lookup[key] = len(out); out.append(p)
        return lookup[key]
    for face in faces:
        polygon = []
        for a, b in zip(face, list(face[1:])+[face[0]]):
            p, q = vertices[a], vertices[b]
            if p[0] >= 0: polygon.append(p)
            if (p[0] < 0) != (q[0] < 0):
                r = p + (-p[0] / (q[0]-p[0])) * (q-p); r[0] = 0.
                polygon.append(r)
        ids = [vertex(p) for p in polygon]
        ids = [a for i,a in enumerate(ids) if a != ids[i-1]]
        if len(ids) >= 3: polygons.append(ids)
    if not polygons: raise ValueError('Surface has no area in X>=0')
    active=sorted({a for face in polygons for a in face});mapping={a:i for i,a in enumerate(active)}
    polygons=[[mapping[a] for a in face] for face in polygons]
    return np.asarray(out)[active], polygons, {'seam_snap_max':snap_max,'seam_snapped_vertices':len(snap), 'seam_tolerance':seam_tolerance,
                                     'input_faces':len(faces), 'clipped_faces':len(polygons)}


class Surface:
    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, float)
        self.faces = np.asarray([[f[0],f[i],f[i+1]] for f in faces for i in range(1,len(f)-1)], int)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or not np.isfinite(self.vertices).all():
            raise ValueError('Invalid surface vertices')
        self.triangles = self.vertices[self.faces]
        self.a = self.triangles[:,0]; self.u = self.triangles[:,1]-self.a; self.v = self.triangles[:,2]-self.a
        self.uu = np.einsum('ij,ij->i',self.u,self.u); self.uvdot = np.einsum('ij,ij->i',self.u,self.v)
        self.vv = np.einsum('ij,ij->i',self.v,self.v); self.det = self.uu*self.vv-self.uvdot**2
        if np.any(self.det <= 1e-20): raise ValueError('Degenerate surface triangle')
        self.uv = None

    def locate(self, points):
        points = np.asarray(points, float).reshape(-1,3)
        if not np.isfinite(points).all(): raise ValueError('Nonfinite projection point')
        result, triangles, barycentrics = [], [], []
        for start in range(0,len(points),128):
            p = points[start:start+128,None,:]; w = p-self.a
            wu = np.einsum('pti,ti->pt',w,self.u); wv = np.einsum('pti,ti->pt',w,self.v)
            b = (wu*self.vv-wv*self.uvdot)/self.det; c = (wv*self.uu-wu*self.uvdot)/self.det
            bary = np.stack((1-b-c,b,c),axis=-1)
            q = self.a + b[:,:,None]*self.u + c[:,:,None]*self.v
            dist = np.sum((p-q)**2,axis=-1); dist[np.min(bary,axis=-1)<0] = np.inf
            for i,j in ((0,1),(1,2),(2,0)):
                a=self.triangles[:,i]; e=self.triangles[:,j]-a
                t=np.clip(np.einsum('pti,ti->pt',p-a,e)/np.sum(e*e,axis=1),0,1)
                edgeq=a+t[:,:,None]*e; edged=np.sum((p-edgeq)**2,axis=-1)
                better=edged<dist
                eb=np.zeros_like(bary); eb[:,:,i]=1-t; eb[:,:,j]=t
                q=np.where(better[:,:,None],edgeq,q); bary=np.where(better[:,:,None],eb,bary);dist=np.minimum(dist,edged)
            idx=np.argmin(dist,axis=1); row=np.arange(len(idx))
            result.extend(q[row,idx]); triangles.extend(idx); barycentrics.extend(bary[row,idx])
        return np.asarray(result).reshape(-1,3), np.asarray(triangles,int), np.asarray(barycentrics).reshape(-1,3)

    closest = locate

    def project(self, points):
        return self.locate(points)[0]

    def parameterize(self):
        """Tutte map of a connected manifold disk, with convex arclength boundary."""
        edges=Counter(tuple(sorted((a,b))) for f in self.faces for a,b in zip(f,np.roll(f,-1)))
        if any(n>2 for n in edges.values()): raise ValueError('Nonmanifold surface edges')
        adj=defaultdict(set); ba=defaultdict(list)
        for a,b in edges:
            adj[a].add(b);adj[b].add(a)
            if edges[(a,b)]==1:ba[a].append(b);ba[b].append(a)
        if not ba or any(len(v)!=2 for v in ba.values()):raise ValueError('Surface boundary is not manifold')
        boundary=[min(ba)];prev=-1
        while True:
            nxt=next(n for n in ba[boundary[-1]] if n!=prev)
            if nxt==boundary[0]:break
            if nxt in boundary:raise ValueError('Surface boundary self-repeats')
            prev=boundary[-1];boundary.append(nxt)
        if len(boundary)!=len(ba) or len(adj)-len(edges)+len(self.faces)!=1:
            raise ValueError('Surface must be a single disk; holes/components are not silently filled')
        lengths=np.linalg.norm(self.vertices[np.roll(boundary,-1)]-self.vertices[boundary],axis=1)
        angles=2*np.pi*np.r_[0,np.cumsum(lengths[:-1])]/sum(lengths)
        uv=np.zeros((len(self.vertices),2));uv[boundary]=np.column_stack((np.cos(angles),np.sin(angles)))
        inside=sorted(set(adj)-set(boundary)); index={a:i for i,a in enumerate(inside)}
        if inside:
            mat=lil_matrix((len(inside),len(inside)));rhs=np.zeros((len(inside),2))
            for a,i in index.items():
                mat[i,i]=len(adj[a])
                for b in adj[a]:
                    if b in index:mat[i,index[b]]=-1
                    else:rhs[i]+=uv[b]
            uv[inside]=spsolve(mat.tocsr(),rhs)
        t=uv[self.faces]; area=(t[:,1,0]-t[:,0,0])*(t[:,2,1]-t[:,0,1])-(t[:,1,1]-t[:,0,1])*(t[:,2,0]-t[:,0,0])
        if np.any(abs(area)<1e-13) or not (np.all(area>0) or np.all(area<0)):
            raise ValueError('Surface parameterization folds or degenerates')
        self.uv=uv;self.boundary=boundary
        return uv

    def to_uv(self,points):
        if self.uv is None:self.parameterize()
        q,idx,bary=self.locate(points)
        return np.einsum('ni,nij->nj',bary,self.uv[self.faces[idx]]),q

    def from_uv(self, points):
        if self.uv is None:self.parameterize()
        points=np.asarray(points,float).reshape(-1,2);t=self.uv[self.faces]
        a=t[:,0];u=t[:,1]-a;v=t[:,2]-a;det=u[:,0]*v[:,1]-u[:,1]*v[:,0]
        output=[]
        for p in points:
            w=p-a;b=(w[:,0]*v[:,1]-w[:,1]*v[:,0])/det;c=(u[:,0]*w[:,1]-u[:,1]*w[:,0])/det
            bary=np.column_stack((1-b-c,b,c));idx=int(np.argmax(np.min(bary,axis=1)))
            if min(bary[idx]) < -1e-6: raise ValueError('Parameter point outside surface disk')
            output.append(bary[idx]@self.triangles[idx])
        return np.asarray(output).reshape(-1,3)
