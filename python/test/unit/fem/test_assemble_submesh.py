# TODO Test replacing mesh with submesh for existing assembler tests

from dolfinx.mesh import (create_unit_square, create_rectangle,
                          locate_entities, create_submesh, GhostMode)
from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx import fem

import ufl

import numpy as np


def assemble(mesh):
    V = fem.FunctionSpace(mesh, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dx = ufl.Measure("dx", domain=mesh)
    a = fem.form(ufl.inner(u, v) * dx)

    A = fem.petsc.assemble_matrix(a)
    A.assemble()

    return A


def test_submesh_cell_assembly():
    n = 2
    mesh = create_unit_square(MPI.COMM_WORLD, n, n)
    A_unit_mesh = assemble(mesh)

    mesh = create_rectangle(
        MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n))
    edim = mesh.topology.dim
    entities = locate_entities(mesh, edim, lambda x: x[0] <= 1.0)
    submesh = create_submesh(mesh, edim, entities)[0]

    A_submesh = assemble(submesh)

    # FIXME Just comparing norms for now, need to communicate to single
    # rank and compare properly
    assert(np.isclose(A_unit_mesh.norm(), A_submesh.norm()))
