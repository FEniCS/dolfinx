from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
import ufl
from basix.ufl import element

avail_partitioners = []
if dolfinx.has_parmetis:
    avail_partitioners.append(dolfinx.graph.partitioner_parmetis)

if dolfinx.has_kahip:
    avail_partitioners.append(dolfinx.graph.partitioner_kahip)

if dolfinx.has_ptscotch:
    avail_partitioners.append(dolfinx.graph.partitioner_scotch)


@pytest.mark.parametrize("partitioner", avail_partitioners)
def test_partitioner(partitioner):
    cells = np.array([[]], dtype=np.int64)
    quad_points = np.array([[0, 0], [0.3, 0]], dtype=np.float64)

    ufl_quad = ufl.Mesh(element("Lagrange", "quadrilateral", 1, shape=(2,)))

    cell_part = dolfinx.mesh.create_cell_partitioner(partitioner())
    dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, quad_points, ufl_quad, partitioner=cell_part)
