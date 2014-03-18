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

if(MPI.size(mpi_comm_world()) == 1):
    print "Only works with MPI"
    quit()

# parameters["mesh_partitioner"] = "ParMETIS"
M = UnitSquareMesh(12, 12)
shared_vertices = M.topology().shared_entities(0).keys()
shared_cells = M.topology().shared_entities(M.topology().dim())

verts_note = []
for k,val in M.topology().shared_entities(0).iteritems():
    vtx = Vertex(M, k)
    verts_note.append( (vtx.point().x(), vtx.point().y(), " "+str(val)) )

x,y = M.coordinates().transpose()

cell_ownership = M.data().array("ghost_owner", M.topology().dim())
process_number = MPI.rank(M.mpi_comm())

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
    xavg = np.mean(np.array(xc))
    yavg = np.mean(np.array(yc))
    cell_str='x'
    if c.index() in shared_cells.keys():
        cell_str = shared_cells[c.index()] 
    cells_note.append((xavg, yavg, cell_str))
    cells_store.append(zip(xc,yc))
    
    colors.append(cmap[cell_ownership[idx]])
    idx += 1

fig, ax = plt.subplots()

# Make the collection and add it to the plot.
coll = PolyCollection(cells_store, facecolors=colors, edgecolors='#cccccc')
ax.add_collection(coll)

plt.plot(x, y, marker='o', color='black', linestyle='none')
plt.plot(x[shared_vertices], y[shared_vertices], marker='o', color='green', linestyle='none')
plt.xlim((-0.1,1.1))
plt.ylim((-0.1,1.1))

for note in cells_note:
    plt.text(note[0], note[1], note[2], verticalalignment='center',
             horizontalalignment='center', size=10)

for note in verts_note:
    plt.text(note[0], note[1], note[2], size=10, verticalalignment='center')

plt.show()

