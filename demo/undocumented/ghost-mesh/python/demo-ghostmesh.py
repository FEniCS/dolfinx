#!/usr/bin/python
#
# very rough demo to test out ghost cells
# run with mpirun 
#
from dolfin import *
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib as mpl
import numpy as np
import sys

parameters["ghost_mode"] = "shared_vertex"

n = 0

if(len(sys.argv) == 2):
    try:
        n = int(sys.argv[1])
    except:
        n = 0

if(MPI.size(mpi_comm_world()) == 1):
    print "Only works with MPI"
    quit()

mpi_rank = MPI.rank(mpi_comm_world())

# parameters["mesh_partitioner"] = "ParMETIS"
M = UnitSquareMesh(5, 5)
M = refine(M)

shared_vertices = M.topology().shared_entities(0).keys()
shared_cells = M.topology().shared_entities(M.topology().dim())

num_regular_vertices = M.topology().size(0) - M.topology().size_ghost(0)

ghost_vertices = range(num_regular_vertices, M.topology().size(0))

verts_note = []
if (n == 0):
    for k,val in M.topology().shared_entities(0).iteritems():
        vtx = Vertex(M, k)
        verts_note.append( (vtx.point().x(), vtx.point().y(), " "+str(val)) )
elif (n == 1):
    for i in range(M.num_vertices()):
        vtx = Vertex(M, i)
        val = vtx.global_index()
        verts_note.append( (vtx.point().x(), vtx.point().y(), " "+str(val)) )
else:
    for i in range(M.num_vertices()):
        vtx = Vertex(M, i)
        val = vtx.index()
        verts_note.append( (vtx.point().x(), vtx.point().y(), " "+str(val)) )

x,y = M.coordinates().transpose()

process_number = MPI.rank(M.mpi_comm())

cell_ownership = np.ones(M.num_cells(),dtype='int')*process_number
cell_owner = M.topology().cell_owner()
if len(cell_owner) > 0 :
    cell_ownership[-len(cell_owner):] = cell_owner

cells_store=[]
cells_note=[]
colors=[]
cmap=['red', 'green', 'yellow', 'purple', 'pink', 'grey', 'blue', 'brown']

idx = 0
for c in cells(M):
    xc=[]
    yc=[]
    for v in vertices(c):
        xc.append(v.point().x())
        yc.append(v.point().y())
    xavg = c.midpoint().x()
    yavg = c.midpoint().y()
    cell_str=str([mpi_rank])
    if c.index() in shared_cells.keys():
        cell_str = str(shared_cells[c.index()])
    else:
        cell_str = ""
    cells_note.append((xavg, yavg, cell_str))
    cells_store.append(zip(xc,yc))
    
    colors.append(cmap[cell_ownership[c.index()]])
    idx += 1

num_regular_facets = M.topology().size(1) - M.topology().size_ghost(1)
facet_note = []
shared_facets = M.topology().shared_entities(1)
for f in facets(M):
    if (n < 3):
        facet_note.append((f.midpoint().x(), f.midpoint().y(), f.global_index()))
    elif (n == 3):
        facet_note.append((f.midpoint().x(), f.midpoint().y(), f.index()))
    else:
        if (f.index() in shared_facets.keys()):
            facet_note.append((f.midpoint().x(), f.midpoint().y(), shared_facets[f.index()]))

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
    plt.text(note[0], note[1], note[2], size=8, verticalalignment='center', backgroundcolor='#eeeeee')

for i in range(num_regular_facets):
    note = facet_note[i]
    plt.text(note[0], note[1], note[2], size=8, verticalalignment='center', backgroundcolor='#80ff80')

plt.show()

