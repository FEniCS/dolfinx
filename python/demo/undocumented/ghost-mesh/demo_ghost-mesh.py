
from dolfin import *
import numpy as np
import sys, os

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
except ImportError:
    print("This demo requires matplotlib! Bye.")
    exit()

n = 0

if(len(sys.argv) == 2):
    try:
        n = int(sys.argv[1])
    except:
        n = 0

if(MPI.size(MPI.comm_world) == 1):
    print("Only works with MPI")
    quit()

#parameters["mesh_partitioner"] = "ParMETIS"

mesh = RectangleMesh.create(MPI.comm_world, [Point(0,0), Point(1, 1)], [8, 8], CellType.Type.triangle, cpp.mesh.GhostMode.shared_vertex)
# mesh = refine(M)

shared_vertices = np.fromiter(mesh.topology.shared_entities(0).keys(), dtype='uintc')
shared_cells = mesh.topology.shared_entities(mesh.topology.dim)

num_regular_vertices = mesh.topology.ghost_offset(0)

ghost_vertices = np.arange(num_regular_vertices, mesh.topology.size(0))

verts_note = []
if (n == 0):
    for k,val in mesh.topology.shared_entities(0).items():
        vtx = Vertex(mesh, k).point().array()
        verts_note.append( (vtx[0], vtx[1], " "+str(val)) )
elif (n == 1):
    for i in range(mesh.num_vertices()):
        vtx = Vertex(mesh, i)
        val = vtx.global_index()
        verts_note.append( (vtx.point().array()[0], vtx.point().array()[1], " "+str(val)) )
else:
    for i in range(mesh.num_vertices()):
        vtx = Vertex(mesh, i)
        val = vtx.index()
        verts_note.append( (vtx.point().array()[0], vtx.point().array()[1], " "+str(val)) )

x,y = mesh.geometry.points.transpose()

rank = MPI.rank(mesh.mpi_comm())

cell_ownership = np.ones(mesh.num_cells(),dtype='int')*rank
cell_owner = mesh.topology.cell_owner()
if len(cell_owner) > 0 :
    cell_ownership[-len(cell_owner):] = cell_owner

cells_store=[]
cells_note=[]
colors=[]
cmap=['red', 'green', 'yellow', 'purple', 'pink', 'grey', 'blue', 'brown']

idx = 0
for c in Cells(mesh, cpp.mesh.MeshRangeType.ALL):
    xc=[]
    yc=[]
    for v in VertexRange(c):
        xc.append(v.point().array()[0])
        yc.append(v.point().array()[1])
    xavg = c.midpoint().array()[0]
    yavg = c.midpoint().array()[1]
    cell_str=str(c.index())
#    if c.index() in shared_cells.keys():
#        cell_str = str(shared_cells[c.index()])
#    else:
#        cell_str = str(c.index())
    cells_note.append((xavg, yavg, cell_str))
    cells_store.append(list(zip(xc,yc)))

    colors.append(cmap[cell_ownership[c.index()]])
    idx += 1

num_regular_facets = mesh.topology.ghost_offset(1)
facet_note = []
shared_facets = mesh.topology.shared_entities(1)
for f in Facets(mesh, cpp.mesh.MeshRangeType.ALL):
    if (f.num_global_entities(2) == 2):
        color='#ffff88'
    else:
        color='#ff88ff'
    if (not f.is_ghost()):
        if (f.num_global_entities(2) == 2):
            color='#ffff00'
        else:
            color='#ff00ff'

    if (n < 3):
        facet_note.append((f.midpoint().array()[0], f.midpoint().array()[1], f.global_index(), color))
    elif (n == 3):
        facet_note.append((f.midpoint().array()[0], f.midpoint().array()[1], f.index(), color))
    else:
        if (f.index() in shared_facets.keys()):
            facet_note.append((f.midpoint().array()[0], f.midpoint().array()[1], shared_facets[f.index()], color))

fig, ax = plt.subplots()

# Make the collection and add it to the plot.
coll = PolyCollection(cells_store, facecolors=colors, edgecolors='#cccccc')
ax.add_collection(coll)

plt.plot(x, y, marker='o', color='black', linestyle='none')
plt.plot(x[shared_vertices], y[shared_vertices], marker='o', color='green', linestyle='none')
plt.plot(x[ghost_vertices], y[ghost_vertices], marker='o', color='yellow', linestyle='none')

xlim = ax.get_xlim()
ylim = ax.get_ylim()

plt.xlim((xlim[0] - 0.1, xlim[1] + 0.1))
plt.ylim((ylim[0] - 0.1, ylim[1] + 0.1))

for note in cells_note:
    plt.text(note[0], note[1], note[2], verticalalignment='center',
             horizontalalignment='center', size=8)

for note in verts_note:
    plt.text(note[0], note[1], note[2], size=8, verticalalignment='center')

for note in facet_note:
    plt.text(note[0], note[1], note[2], size=8, verticalalignment='center', backgroundcolor=note[3])

# Q = MeshFunction("double", mesh, mesh.topology.dim-1)

# # Save solution in XDMF format if available
# xdmf = XDMFFile(mesh.mpi_comm(), "Q.xdmf")
# if config.has_hdf5:
#     xdmf.write(Q)
# elif MPI.size(mesh.mpi_comm()) == 1:
#     encoding = XDMFFile.Encoding.ASCII
#     xdmf.write(Q, encoding)
# else:
#     # Save solution in vtk format
#     xdmf = File("Q.pvd")
#     xdmf << Q

plt.savefig("mesh-rank%d.png" % rank)
if os.environ.get("DOLFIN_NOPLOT", "0") == "0":
    plt.show()
