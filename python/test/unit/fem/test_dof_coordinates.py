import numpy as np
import pytest

from dolfinx.fem import Function, FunctionSpace
from dolfinx.generation import UnitCubeMesh, UnitSquareMesh

from mpi4py import MPI
from petsc4py import PETSc


@pytest.mark.parametrize("degree", range(1, 5))
def test_dof_coords_2d(degree):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 10, 10)
    V = FunctionSpace(mesh, ("Lagrange", degree))
    u = Function(V)

    u.interpolate(lambda x: x[0])
    x = V.tabulate_dof_coordinates()
    val = u.x.array
    for i in range(len(val)):
        assert np.isclose(x[i, 0], val[i], rtol=1e-3)


@pytest.mark.parametrize("degree", range(1, 5))
def test_dof_coords_3d(degree):
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 10, 10, 10)
    V = FunctionSpace(mesh, ("Lagrange", degree))
    u = Function(V)

    u.interpolate(lambda x: x[0])
    x = V.tabulate_dof_coordinates()
    val = u.x.array
    for i in range(len(val)):
        assert np.isclose(x[i, 0], val[i], rtol=1e-3)
