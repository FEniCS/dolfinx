"""Unit tests for the refinement library"""

# Copyright (C) 2014 Chris Richardson
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
# First added:  2011-08-23
# Last changed:

import unittest
from dolfin import *

class MarkerMeshRefinement(unittest.TestCase):

    def test_marker_refine2D(self):
        mesh = UnitSquareMesh(6, 6)
        for i in range(4):
            marker = CellFunction("bool", mesh, False)
            for c in cells(mesh):
                p = c.midpoint()
                x = p.x() - 0.5
                y = p.y() - 0.5
                r = sqrt(x*x + y*y)
                if (r < 0.25):
                    marker[c] = True

            mesh = refine(mesh, marker, False)

            vtot = 0.0
            for c in cells(mesh):
                vtot += c.volume()
            vtot = MPI.sum(mesh.mpi_comm(), vtot)
            self.assertAlmostEqual(vtot, 1.0)


    def test_marker_refine3D(self):
        mesh = UnitCubeMesh(6, 6, 6)
        for i in range(4):
            marker = CellFunction("bool", mesh, False)
            for c in cells(mesh):
                p = c.midpoint()
                x = p.x() - 0.5
                y = p.y() - 0.5
                z = p.z() - 0.5
                r = sqrt(x*x + y*y + z*z)
                if (r < 0.15):
                    marker[c] = True

            mesh = refine(mesh, marker, False)

            vtot = 0.0
            for c in cells(mesh):
                vtot += c.volume()
            vtot = MPI.sum(mesh.mpi_comm(), vtot)
            self.assertAlmostEqual(vtot, 1.0)

if __name__ == "__main__":
    unittest.main()
