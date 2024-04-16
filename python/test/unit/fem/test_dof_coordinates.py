from mpi4py import MPI

import numpy as np
import pytest

from dolfinx.fem import Function, functionspace
from dolfinx.mesh import create_unit_cube, create_unit_square


@pytest.mark.parametrize("degree", range(1, 5))
def test_dof_coords_2d(degree):
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = functionspace(mesh, ("Lagrange", degree))
    u = Function(V)
    u.interpolate(lambda x: x[0])
    u.x.scatter_forward()
    x = V.tabulate_dof_coordinates()
    val = u.x.array
    for i in range(len(val)):
        assert np.isclose(x[i, 0], val[i], rtol=1e-3)


@pytest.mark.parametrize("degree", range(1, 5))
def test_dof_coords_3d(degree):
    mesh = create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    V = functionspace(mesh, ("Lagrange", degree))
    u = Function(V)
    u.interpolate(lambda x: x[0])
    u.x.scatter_forward()
    x = V.tabulate_dof_coordinates()
    val = u.x.array
    for i in range(len(val)):
        assert np.isclose(x[i, 0], val[i], rtol=1e-3)
