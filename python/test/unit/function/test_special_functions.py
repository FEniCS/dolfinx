# Copyright (C) 2011 Kristian B. Oelgaard
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the function library"""

import dolfinx
import pytest
import ufl
from dolfinx import UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh
from dolfinx_utils.test.skips import skip_in_parallel
from mpi4py import MPI


@pytest.mark.skip
@skip_in_parallel
def testFacetArea():
    references = [(UnitIntervalMesh(MPI.COMM_WORLD, 1), 2, 2), (UnitSquareMesh(
        MPI.COMM_WORLD, 1, 1), 4, 4), (UnitCubeMesh(MPI.COMM_WORLD, 1, 1, 1),
                                       6, 3)]
    for mesh, surface, ref_int in references:
        c0 = ufl.FacetArea(mesh)
        c1 = dolfinx.FacetArea(mesh)
        assert (c0)
        assert (c1)


#        assert round(assemble(c*dx(mesh)) - 1, 7) == 0
#        assert round(assemble(c*ds(mesh)) - surface, 7) == 0
#        assert round(assemble(c0*ds(mesh)) - ref_int, 7) == 0
#        assert round(assemble(c1*ds(mesh)) - ref_int, 7) == 0
