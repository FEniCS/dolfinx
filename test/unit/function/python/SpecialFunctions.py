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
# Last changed: 2011-11-28

import unittest
from dolfin import *

class SpecialFunctions(unittest.TestCase):

    def testFacetArea(self):
        if MPI.num_processes() == 1:
            references = [(UnitIntervalMesh(1), 2, 2),\
                          (UnitSquareMesh(1,1), 4, 4),\
                          (UnitCubeMesh(1,1,1), 6, 3)]
            for mesh, surface, ref_int in references:
                c = Constant(1, mesh.ufl_cell())
                c0 = mesh.ufl_cell().facet_area
                c1 = FacetArea(mesh)
                self.assertAlmostEqual(assemble(c*dx, mesh=mesh), 1)
                self.assertAlmostEqual(assemble(c*ds, mesh=mesh), surface)
                self.assertAlmostEqual(assemble(c0*ds, mesh=mesh), ref_int)
                self.assertAlmostEqual(assemble(c1*ds, mesh=mesh), ref_int)

if __name__ == "__main__":
    unittest.main()
