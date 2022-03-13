# TODO Test replacing mesh with submesh for existing assembler tests

from dolfinx.mesh import (create_unit_square, create_rectangle,
                          locate_entities, create_submesh, GhostMode)
from mpi4py import MPI
from dolfinx import fem

import ufl

import numpy as np

import pytest


def assemble(mesh):
    V = fem.FunctionSpace(mesh, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dx = ufl.Measure("dx", domain=mesh)
    a = fem.form(ufl.inner(u, v) * dx)

    A = fem.petsc.assemble_matrix(a)
    A.assemble()

    return A


# TODO Try different ghost modes
@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
def test_submesh_cell_assembly(n, ghost_mode):
    mesh = create_unit_square(MPI.COMM_WORLD, n, n,
                              ghost_mode=ghost_mode)
    A_unit_mesh = assemble(mesh)

    mesh = create_rectangle(
        MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
        ghost_mode=ghost_mode)
    edim = mesh.topology.dim
    entities = locate_entities(mesh, edim, lambda x: x[0] <= 1.0)
    submesh = create_submesh(mesh, edim, entities)[0]

    A_submesh = assemble(submesh)

    # FIXME Would probably be better to compare entries rather than just
    # norms
    assert(np.isclose(A_unit_mesh.norm(), A_submesh.norm()))
