"""Unit tests for the function library"""

# Copyright (C) 2011 Kristian B. Oelgaard
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2011-11-28
# Last changed: 2014-05-30

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
