"""Unit tests for the function library"""

# Copyright (C) 2011 Kristian B. Oelgaard
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
from dolfin import *
import ufl
import dolfin
from dolfin_utils.test import skip_in_parallel

@skip_in_parallel
def testFacetArea():

    references = [(UnitIntervalMesh(1), 2, 2),\
                  (UnitSquareMesh(1,1), 4, 4),\
                  (UnitCubeMesh(1,1,1), 6, 3)]
    for mesh, surface, ref_int in references:
        c = Constant(1, mesh.ufl_cell()) # FIXME
        c0 = ufl.FacetArea(mesh)
        c1 = dolfin.FacetArea(mesh)
        assert round(assemble(c*dx(mesh)) - 1, 7) == 0
        assert round(assemble(c*ds(mesh)) - surface, 7) == 0
        assert round(assemble(c0*ds(mesh)) - ref_int, 7) == 0
        assert round(assemble(c1*ds(mesh)) - ref_int, 7) == 0
