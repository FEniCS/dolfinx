import matplotlib.pyplot as plt
import dolfinx.plotting
import time
from IPython import embed
import dolfinx
import dolfinx.geometry
import numpy as np
from mpi4py import MPI

mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 2, 2)


imap = mesh.topology.index_map(mesh.topology.dim)
num_cells = imap.size_local + imap.num_ghosts
start = time.time()
cells = [2, 1, 6, 5]  # range(num_cells)

tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim, cells)
midpoint_tree = dolfinx.geometry.create_midpoint_tree(mesh, mesh.topology.dim, cells)
print(midpoint_tree.get_bbox(0))
end = time.time()
tot = end - start
print("Midpointtree init (avg) {0:.2e}".format(MPI.COMM_WORLD.allreduce(tot, op=MPI.SUM) / MPI.COMM_WORLD.size))
midpoints = dolfinx.cpp.mesh.midpoints(mesh, mesh.topology.dim, cells)
ppp = np.array([[0.36, 0.6, 0]])
for mid in midpoints:
    print(mid, np.linalg.norm(mid - ppp)**2)
entity = dolfinx.cpp.geometry.compute_closest_entity(tree, midpoint_tree, mesh, ppp)

print("ANSWER", entity)


dolfinx.plotting.plot(mesh)

plt.savefig("mesh.png")
