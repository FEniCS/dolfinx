from mpi4py import MPI
import basix.ufl
import ufl
import dolfinx
import numpy as np
import pytest


@pytest.mark.parametrize("degree", range(1, 4))
def test_transpose(degree):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    for symmetry in [True, False]:
        e = basix.ufl.element("Lagrange", "triangle", degree, shape=(2, 2), symmetry=symmetry)

        space = dolfinx.fem.functionspace(mesh, e)

        f = dolfinx.fem.Function(space)
        f.interpolate(lambda x: (x[0], x[1], x[0] ** 3, x[0]))

        form = dolfinx.fem.form(ufl.inner(f - ufl.transpose(f), f - ufl.transpose(f)) * ufl.dx)
        assert np.isclose(dolfinx.fem.assemble_scalar(form), 0) == symmetry
