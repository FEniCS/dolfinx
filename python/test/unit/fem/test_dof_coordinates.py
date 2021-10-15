import pytest
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np


@pytest.mark.parametrize("degree", range(1, 5))
def test_dof_coords_2d(degree):
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 10, 10)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", degree))
    u = dolfinx.Function(V)

    u.interpolate(lambda x: x[0])
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    x = V.tabulate_dof_coordinates()
    val = u.vector.array
    for i in range(len(val)):
        assert np.isclose(x[i, 0], val[i], rtol=1e-3)


@pytest.mark.parametrize("degree", range(1, 5))
def test_dof_coords_3d(degree):
    mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 10, 10, 10)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", degree))
    u = dolfinx.Function(V)

    u.interpolate(lambda x: x[0])
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    x = V.tabulate_dof_coordinates()
    val = u.vector.array
    for i in range(len(val)):
        assert np.isclose(x[i, 0], val[i], rtol=1e-3)
