import dolfinx
from mpi4py import MPI
import numpy as np
from IPython import embed

mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 1, 1)
tdim = mesh.topology.dim
num_cells = mesh.topology.index_map(tdim).size_local
indices = np.arange(num_cells, dtype=np.int32)
values = indices
ct = dolfinx.MeshTags(mesh, tdim, indices[0:1], values[0:1])
V = dolfinx.FunctionSpace(ct, ("CG", 1))
# embed()