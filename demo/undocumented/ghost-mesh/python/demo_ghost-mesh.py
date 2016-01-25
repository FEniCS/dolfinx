#!/usr/bin/python
#
# very rough demo to test out ghost cells
# run with mpirun
#
from __future__ import print_function

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
except ImportError:
    print("matplotib is needed to run this demo. Import was not "
          "succesful. Exiting gracefully.")
    exit()

from dolfin import *

import numpy as np
import sys
import six

parameters["ghost_mode"] = "shared_vertex"
#parameters["ghost_mode"] = "shared_facet"
#parameters["ghost_mode"] = "None"
parameters["reorder_cells_gps"] = True
parameters["reorder_vertices_gps"] = True

n = 0

if(len(sys.argv) == 2):
    try:
        n = int(sys.argv[1])
    except:
        n = 0

if(MPI.size(mpi_comm_world()) == 1):
    print("Only works with MPI")
    quit()

#parameters["mesh_partitioner"] = "ParMETIS"

mesh = UnitSquareMesh(8, 8)
# mesh = refine(M)

shared_vertices = mesh.topology().shared_entities(0).keys()
shared_cells = mesh.topology().shared_entities(mesh.topology().dim())

num_regular_vertices = mesh.topology().ghost_offset(0)

ghost_vertices = range(num_regular_vertices, mesh.topology().size(0))

verts_note = []
if (n == 0):
    for k,val in six.iteritems(mesh.topology().shared_entities(0)):
        vtx = Vertex(mesh, k)
        verts_note.append( (vtx.point().x(), vtx.point().y(), " "+str(val)) )
elif (n == 1):
    for i in range(mesh.num_vertices()):
        vtx = Vertex(mesh, i)
        val = vtx.global_index()
        verts_note.append( (vtx.point().x(), vtx.point().y(), " "+str(val)) )
else:
    for i in range(mesh.num_vertices()):
        vtx = Vertex(mesh, i)
        val = vtx.index()
        verts_note.append( (vtx.point().x(), vtx.point().y(), " "+str(val)) )

x,y = mesh.coordinates().transpose()

rank = MPI.rank(mesh.mpi_comm())

cell_ownership = np.ones(mesh.num_cells(),dtype='int')*rank
cell_owner = mesh.topology().cell_owner()
if len(cell_owner) > 0 :
    cell_ownership[-len(cell_owner):] = cell_owner

cells_store=[]
cells_note=[]
colors=[]
cmap=['red', 'green', 'yellow', 'purple', 'pink', 'grey', 'blue', 'brown']

idx = 0
for c in cells(mesh, "all"):
    xc=[]
    yc=[]
    for v in vertices(c):
        xc.append(v.point().x())
        yc.append(v.point().y())
    xavg = c.midpoint().x()
    yavg = c.midpoint().y()
    cell_str=str(c.index())
#    if c.index() in shared_cells.keys():
#        cell_str = str(shared_cells[c.index()])
#    else:
#        cell_str = str(c.index())
    cells_note.append((xavg, yavg, cell_str))
    cells_store.append(zip(xc,yc))

    colors.append(cmap[cell_ownership[c.index()]])
    idx += 1

num_regular_facets = mesh.topology().ghost_offset(1)
facet_note = []
shared_facets = mesh.topology().shared_entities(1)
for f in facets(mesh, "all"):
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
        facet_note.append((f.midpoint().x(), f.midpoint().y(), f.global_index(), color))
    elif (n == 3):
        facet_note.append((f.midpoint().x(), f.midpoint().y(), f.index(), color))
    else:
        if (f.index() in shared_facets.keys()):
            facet_note.append((f.midpoint().x(), f.midpoint().y(), shared_facets[f.index()], color))

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

# Q = FacetFunction("double", mesh)

# # Save solution in XDMF format if available
# xdmf = XDMFFile(mesh.mpi_comm(), "Q.xdmf")
# if has_hdf5():
#     xdmf.write(Q)
# elif MPI.size(mesh.mpi_comm()) == 1:
#     encoding = XDMFFile.Encoding_ASCII
#     xdmf.write(Q, encoding)
# else:
#     # Save solution in vtk format
#     xdmf = File("Q.pvd")
#     xdmf << Q

plt.savefig("mesh-rank%d.png" % rank)
