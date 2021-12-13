import numpy as np
import pytest

from dolfinx.fem import Function, FunctionSpace
from dolfinx.mesh import create_unit_cube_mesh, create_unit_square_mesh

from mpi4py import MPI
from petsc4py import PETSc


@pytest.mark.parametrize("degree", range(1, 5))
def test_dof_coords_2d(degree):
    mesh = create_unit_square_mesh(MPI.COMM_WORLD, 10, 10)
    V = FunctionSpace(mesh, ("Lagrange", degree))
    u = Function(V)

    u.interpolate(lambda x: x[0])
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    x = V.tabulate_dof_coordinates()
    val = u.vector.array
    for i in range(len(val)):
        assert np.isclose(x[i, 0], val[i], rtol=1e-3)


@pytest.mark.parametrize("degree", range(1, 5))
def test_dof_coords_3d(degree):
    mesh = create_unit_cube_mesh(MPI.COMM_WORLD, 10, 10, 10)
    V = FunctionSpace(mesh, ("Lagrange", degree))
    u = Function(V)

    u.interpolate(lambda x: x[0])
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    x = V.tabulate_dof_coordinates()
    val = u.vector.array
    for i in range(len(val)):
        assert np.isclose(x[i, 0], val[i], rtol=1e-3)
