"""Unit tests for intersection computation"""

# Copyright (C) 2013 Anders Logg
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
# First added:  2013-12-09
# Last changed: 2013-12-09

import unittest

from dolfin import *

class Issues(unittest.TestCase):

    def test_issue_97(self):
        "Test from Mikael Mortensen (issue #97)"

        N = 2
        L = 1000
        mesh = BoxMesh(0, 0, 0, L, L, L, N, N, N)

        if MPI.size(mesh.mpi_comm()) > 1:
            return

        V = FunctionSpace(mesh, 'CG', 1)
        v = interpolate(Expression('x[0]'), V)
        x = Point(0.5*L, 0.5*L, 0.5*L)
        vx = v(x)

    def test_issue_168(self):
        "Test from Torsten Wendav (issue #168)"

        mesh = UnitCubeMesh(14, 14, 14)
        if MPI.size(mesh.mpi_comm()) > 1:
            return
        V = FunctionSpace(mesh, "Lagrange", 1)
        v = Function(V)
        x = (0.75, 0.25, 0.125)
        vx = v(x)

if __name__ == "__main__":
    print ""
    print "Testing issues reported by users"
    print "------------------------------------------------"
    unittest.main()
