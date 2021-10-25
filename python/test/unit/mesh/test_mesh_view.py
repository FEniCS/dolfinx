# TODO When finished, add to test_mesh.py since "meshview" is a mesh

import dolfinx
from mpi4py import MPI
import numpy as np


def boundary(x):
    lr = np.logical_or(np.isclose(x[0], 0.0),
                       np.isclose(x[0], 1.0))
    tb = np.logical_or(np.isclose(x[1], 0.0),
                       np.isclose(x[1], 1.0))
    return np.logical_or(lr, tb)


n = 1
mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, n, n)
entity_dim = mesh.topology.dim - 1
entities = dolfinx.mesh.locate_entities_boundary(mesh, entity_dim, boundary)

submesh = mesh.sub(entity_dim, entities)
print(submesh)
