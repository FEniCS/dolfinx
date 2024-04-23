from mpi4py import MPI

import numpy as np
import pytest

import basix.ufl
import dolfinx
import ufl


@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("symmetry", [True, False])
def test_transpose(degree, symmetry):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    e = basix.ufl.element("Lagrange", "triangle", degree, shape=(2, 2), symmetry=symmetry)

    space = dolfinx.fem.functionspace(mesh, e)

    f = dolfinx.fem.Function(space)
    f.interpolate(lambda x: (x[0], x[1], x[0] ** 3, x[0]))

    form = dolfinx.fem.form(ufl.inner(f - ufl.transpose(f), f - ufl.transpose(f)) * ufl.dx)
    assert np.isclose(dolfinx.fem.assemble_scalar(form), 0) == symmetry


def test_interpolation():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    def tensor(x):
        mat = np.array([[0], [1], [2], [1], [3], [4], [2], [4], [5]])
        return np.broadcast_to(mat, (9, x.shape[1]))

    element = basix.ufl.element("DG", mesh.basix_cell(), 0, shape=(3, 3))
    symm_element = basix.ufl.element("DG", mesh.basix_cell(), 0, shape=(3, 3), symmetry=True)
    space = dolfinx.fem.functionspace(mesh, element)
    symm_space = dolfinx.fem.functionspace(mesh, symm_element)
    f = dolfinx.fem.Function(space)
    symm_f = dolfinx.fem.Function(symm_space)

    f.interpolate(lambda x: tensor(x))
    symm_f.interpolate(lambda x: tensor(x))

    l2_error = dolfinx.fem.assemble_scalar(dolfinx.fem.form((f - symm_f) ** 2 * ufl.dx))
    atol = 10 * np.finfo(dolfinx.default_scalar_type).resolution
    assert np.isclose(l2_error, 0.0, atol=atol)


def test_eval():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    mat = np.array([0, 1, 2, 1, 3, 4, 2, 4, 5])

    def tensor(x):
        return np.broadcast_to(mat.reshape((9, 1)), (9, x.shape[1]))

    element = basix.ufl.element("DG", mesh.basix_cell(), 0, shape=(3, 3), symmetry=True)
    space = dolfinx.fem.functionspace(mesh, element)
    f = dolfinx.fem.Function(space)

    f.interpolate(lambda x: tensor(x))

    value = f.eval([[0, 0, 0]], [0])

    atol = 10 * np.finfo(dolfinx.default_scalar_type).resolution
    assert np.allclose(value, mat, atol=atol)
