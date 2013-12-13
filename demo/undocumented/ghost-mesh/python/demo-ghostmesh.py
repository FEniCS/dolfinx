#!/usr/bin/python
#
# very rough demo to test out ghost cells
# run with mpirun 
#
from dolfin import *
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib as mpl

if(MPI.num_processes() == 1):
    print "Only works with MPI"
    quit()

# parameters["mesh_partitioner"] = "ParMETIS"
M = UnitSquareMesh(25, 25)
shared_vertices = M.topology().shared_entities(0).keys()

x,y = M.coordinates().transpose()

cmask_array = M.data().array("ghost_mask", M.topology().dim())
vmask_array = M.data().array("ghost_mask", 0) == 0
vmask_array_g = M.data().array("ghost_mask", 0) == 1

cells_store=[]
colors=[]

idx = 0
for c in cells(M):
    xc=[]
    yc=[]
    for v in vertices(c):
        xc.append(v.point().x())
        yc.append(v.point().y())
    cells_store.append(zip(xc,yc))
    
    if(cmask_array[idx] == 0):
        colors.append('#ff0000')
    else:
        colors.append('#c0c0ff')
    idx += 1

fig, ax = plt.subplots()

# Make the collection and add it to the plot.
coll = PolyCollection(cells_store, facecolors=colors, edgecolors='black')
ax.add_collection(coll)

plt.plot(x[vmask_array], y[vmask_array], marker='o', color='red', linestyle='none')
plt.plot(x[shared_vertices], y[shared_vertices], marker='o', color='green', linestyle='none')
plt.plot(x[vmask_array_g], y[vmask_array_g], marker='o', color='blue', linestyle='none')
plt.xlim((-0.1,1.1))
plt.ylim((-0.1,1.1))

plt.show()

