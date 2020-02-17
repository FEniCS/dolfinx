import dolfinx
import ufl
import numpy as np


def test_locate_dofs_geometric():
    mesh = dolfinx.generation.UnitSquareMesh(dolfinx.MPI.comm_world, 4, 8)
    p0, p1 = 1, 2
    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)

    W = dolfinx.function.FunctionSpace(mesh, P0 * P1)
    V = W.sub(0).collapse()

    dofs = dolfinx.fem.locate_dofs_geometrical(
        (W.sub(0), V), lambda x: np.isclose(x.T, [0, 0, 0]).all(axis=1))

    # Check only one dof pair is returned
    assert len(dofs) == 1
    coords_W = W.tabulate_dof_coordinates()
    # Check correct dof returned in W
    assert np.isclose(coords_W[dofs[0][0]], [0, 0, 0]).all()
    # Check correct dof returned in V
    coords_V = V.tabulate_dof_coordinates()
    assert np.isclose(coords_V[dofs[0][1]], [0, 0, 0]).all()
