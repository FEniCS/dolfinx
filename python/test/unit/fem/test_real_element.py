# Copyright (C) 2022 Matthew W. Scroggs
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest

import dolfinx
import ufl
from dolfinx.fem import FunctionSpace, form
from dolfinx.mesh import (CellType, GhostMode, create_unit_cube,
                          create_unit_square, create_unit_interval)
from dolfinx_utils.test.skips import skip_in_parallel

from mpi4py import MPI


@skip_in_parallel
@pytest.mark.parametrize("cell", ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"])
def test_real_element(cell):
    ufl_cell = getattr(ufl, cell)
    dolfinx_cell = getattr(CellType, cell)
    if cell == "interval":
        mesh = create_unit_interval(MPI.COMM_WORLD, 3, GhostMode.shared_facet)
    elif cell.endswith("hedron"):
        mesh = create_unit_cube(MPI.COMM_WORLD, 3, 2, 1, dolfinx_cell, GhostMode.shared_facet)
    else:
        mesh = create_unit_square(MPI.COMM_WORLD, 3, 2, dolfinx_cell, GhostMode.shared_facet)

    U_el = ufl.FiniteElement("R", ufl_cell, 0)
    U = FunctionSpace(mesh, U_el)
    u = ufl.TrialFunction(U)
    v = ufl.TestFunction(U)
    a = form(ufl.inner(u, v) * ufl.dx)

    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()

    assert A.getSize()[0] == A.getSize()[1] == 1
