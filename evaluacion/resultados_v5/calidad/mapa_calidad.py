"""Mapa de calidad por cara de una jaula (18-nov): verde = buena (proporcion <= 2 y sus 4 vertices de valencia 4 o en borde/crease),
amarillo = aceptable (<= 3.5), rojo = estirada (> 3.5). Uso: blender -b -P mapa_calidad.py -- <obj> <salida_png_prefijo>"""
import bpy, sys, os, math
from mathutils import Vector
a = sys.argv[sys.argv.index("--")+1:]
OBJ, OUT = a[0], a[1]
V=[];F=[]
for l in open(OBJ):
    if l.startswith('v '): V.append(tuple(float(x) for x in l.split()[1:4]))
    elif l.startswith('f '): F.append([int(x.split('/')[0])-1 for x in l.split()[1:]])
val=[0]*len(V); ed={}
for f in F:
    for k in range(len(f)):
        e=tuple(sorted((f[k],f[(k+1)%len(f)]))); ed[e]=ed.get(e,0)+1
for (i,j) in ed: val[i]+=1; val[j]+=1
borde={v for e,c in ed.items() if c==1 for v in e}
bpy.ops.wm.read_factory_settings(use_empty=True)
me=bpy.data.meshes.new("m"); me.from_pydata(V,[],F); me.update()
cols=[]; n=[0,0,0]
for f in F:
    P=[Vector(V[i]) for i in f]; L=[(P[k]-P[(k+1)%len(P)]).length for k in range(len(P))]
    r=max(L)/max(min(L),1e-9)
    reg=all(val[i]==4 or i in borde for i in f)
    if r<=2 and reg: c=(0.15,0.7,0.25,1); n[0]+=1
    elif r<=3.5: c=(0.95,0.8,0.2,1); n[1]+=1
    else: c=(0.85,0.15,0.15,1); n[2]+=1
    cols.append(c)
att=me.color_attributes.new("q","FLOAT_COLOR","CORNER")
for p in me.polygons:
    for li in p.loop_indices: att.data[li].color=cols[p.index]
ob=bpy.data.objects.new("o",me); bpy.context.scene.collection.objects.link(ob)
w=ob.copy(); w.data=me.copy(); bpy.context.scene.collection.objects.link(w); m=w.modifiers.new("wf","WIREFRAME"); m.thickness=0.2; m.use_replace=True; m.use_even_offset=False; w.color=(0.1,0.1,0.1,1)

sc=bpy.context.scene; sc.render.engine="BLENDER_WORKBENCH"; sh=sc.display.shading; sh.show_cavity=False
sh.light="STUDIO"; sh.color_type="VERTEX"
sc.render.resolution_x=1600; sc.render.resolution_y=1100
pts=[Vector(v) for v in V]; mn=Vector((min(p.x for p in pts),min(p.y for p in pts),min(p.z for p in pts))); mx=Vector((max(p.x for p in pts),max(p.y for p in pts),max(p.z for p in pts)))
cen=(mn+mx)/2; size=max(mx-mn)
cd=bpy.data.cameras.new("c"); cd.type="ORTHO"; cd.ortho_scale=size*0.95; cd.clip_end=size*10
cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
for nom,dv in {"34":Vector((1,-1.1,0.8)),"arriba":Vector((0,0,1)),"lado":Vector((1,0,0.05))}.items():
    dv.normalize(); cam.location=cen+dv*size*3
    cam.rotation_euler=(cen-cam.location).to_track_quat("-Z","Y").to_euler()
    sc.render.filepath=f"{OUT}_{nom}.png"; bpy.ops.render.render(write_still=True)
t=len(F); print("CALIDAD", OBJ, t, "buenas %.1f%% aceptables %.1f%% estiradas %.1f%%"%tuple(100*x/t for x in n))
