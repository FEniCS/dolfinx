from IPython import embed
import dolfinx
from mpi4py import MPI
import numpy as np

mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 3, 2)
vertex_map = mesh.topology.index_map(0)
s_l = vertex_map.size_local
num_ghosts = len(vertex_map.ghosts)
comm = MPI.COMM_WORLD
entities = []
for i in range(s_l):
    if i % 2 == 0:
        entities.append(i)
for i in range(num_ghosts):
    if i % 2 == 0:
        entities.append(s_l + i)

entities = np.array(entities, dtype=np.int32)
print(comm.rank, entities, s_l)
new_map = dolfinx.cpp.common.compress_index_map(vertex_map, entities)
