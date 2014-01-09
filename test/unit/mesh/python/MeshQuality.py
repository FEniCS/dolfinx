"Unit tests for the MeshQuality class"

# Copyright (C) 2013 Garth N. Wells
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
# First added:  2013-10-07
# Last changed:

import unittest
import numpy
from dolfin import *

class MeshQualityTest(unittest.TestCase):

    def test_radius_ratio_triangle(self):

        # Create mesh and compute rations
        mesh = UnitSquareMesh(12, 12)
        ratios = MeshQuality.radius_ratios(mesh)
        for c in cells(mesh):
            self.assertAlmostEqual(ratios[c], 0.828427124746)

    def test_radius_ratio_tetrahedron(self):

        # Create mesh and compute ratios
        mesh = UnitCubeMesh(14, 14, 14)
        ratios = MeshQuality.radius_ratios(mesh)
        for c in cells(mesh):
            self.assertAlmostEqual(ratios[c], 0.717438935214)
            #print ratio[c]

    def test_radius_ratio_triangle_min_max(self):

        # Create mesh, collpase and compute min ratio
        mesh = UnitSquareMesh(12, 12)

        rmin, rmax = MeshQuality.radius_ratio_min_max(mesh)
        self.assertTrue(rmax <= rmax)

        x = mesh.coordinates()
        x[:, 0] *= 0.0
        rmin, rmax = MeshQuality.radius_ratio_min_max(mesh)
        self.assertAlmostEqual(rmin, 0.0)
        self.assertAlmostEqual(rmax, 0.0)

    def test_radius_ratio_tetrahedron_min_max(self):

        # Create mesh, collpase and compute min ratio
        mesh = UnitCubeMesh(12, 12, 12)

        rmin, rmax = MeshQuality.radius_ratio_min_max(mesh)
        self.assertTrue(rmax <= rmax)

        x = mesh.coordinates()
        x[:, 0] *= 0.0
        rmin, rmax = MeshQuality.radius_ratio_min_max(mesh)
        self.assertAlmostEqual(rmax, 0.0)
        self.assertAlmostEqual(rmax, 0.0)

    def test_radius_ratio_matplotlib(self):

        # Create mesh, collpase and compute min ratio
        mesh = UnitCubeMesh(12, 12, 12)
        test = MeshQuality.radius_ratio_matplotlib_histogram(mesh, 5)
        print test


if MPI.num_processes() == 1:
    class CellRadii(unittest.TestCase):

        def setUp(self):
            # Create 1D mesh with degenerate cell
            self.mesh1d = UnitIntervalMesh(4)
            self.mesh1d.coordinates()[4] = self.mesh1d.coordinates()[3]

            # Create 2D mesh with one equilateral triangle
            self.mesh2d = UnitSquareMesh(1, 1, 'left')
            self.mesh2d.coordinates()[3] += 0.5*(sqrt(3.0)-1.0)

            # Create 3D mesh with regular tetrahedron and degenerate cells
            self.mesh3d = UnitCubeMesh(1, 1, 1)
            self.mesh3d.coordinates()[2][0] = 1.0
            self.mesh3d.coordinates()[7][1] = 0.0

        def test_radius_ratio_min_radius_ratio_max(self):
            rmin, rmax = MeshQuality.radius_ratio_min_max(self.mesh1d)
            self.assertAlmostEqual(rmin, 0.0)
            self.assertAlmostEqual(rmax, 1.0)

            rmin, rmax = MeshQuality.radius_ratio_min_max(self.mesh2d)
            self.assertAlmostEqual(rmin, 2.0*sqrt(2.0)/(2.0+sqrt(2.0)) )
            self.assertAlmostEqual(rmax, 1.0)

            rmin, rmax = MeshQuality.radius_ratio_min_max(self.mesh3d)
            self.assertAlmostEqual(rmin, 0.0)
            self.assertAlmostEqual(rmax, 1.0)

if __name__ == "__main__":
    unittest.main()
