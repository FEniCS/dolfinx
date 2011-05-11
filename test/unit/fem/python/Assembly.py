"""Unit tests for assembly"""

# Copyright (C) 2011 Garth N. Wells
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2011-03-12
# Last changed: 2011-03-12

import unittest
import numpy
from dolfin import *

class Assembly(unittest.TestCase):

    def test_functional_assembly(self):

        mesh = UnitSquare(24, 24)

        # This is a hack to get around a DOLFIN bug
        if MPI.num_processes() > 1:
            cpp.MeshPartitioning.number_entities(mesh, mesh.topology().dim() - 1);

        f = Constant(1.0)
        M = f*dx
        parameters["num_threads"] = 0
        self.assertAlmostEqual(assemble(M, mesh=mesh), 1.0)

        M = f*ds
        parameters["num_threads"] = 0
        self.assertAlmostEqual(assemble(M, mesh=mesh), 4.0)

if __name__ == "__main__":
    print ""
    print "Testing basic DOLFIN assembly operations"
    print "------------------------------------------------"
    unittest.main()
