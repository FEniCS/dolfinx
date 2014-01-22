"""Unit tests for the X3D io library"""

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
# First added:  2013-05-12
# Last changed:

import unittest
from dolfin import *

class X3D_Mesh(unittest.TestCase):
    """Test output of Mesh to X3D file"""

    def test_save_mesh1D(self):
        mesh = UnitIntervalMesh(16)
        file = File("mesh1D.x3d")
        #self.assertRaises(RuntimeError, file << mesh)

    def test_save_mesh2D(self):
        mesh = UnitSquareMesh(16, 16)
        file = File("mesh2D.x3d")
        file << mesh

    def test_save_mesh3D(self):
        mesh = UnitCubeMesh(16, 16, 16)
        file = File("mesh3D.x3d")
        file << mesh

class X3D_MeshFunction(unittest.TestCase):

    def test_save_cell_meshfunction2D(self):
        mesh = UnitSquareMesh(16, 16)
        mf = CellFunction("size_t", mesh, 12)
        file = File("cell_mf2D.x3d")
        file << mf

    def test_save_facet_meshfunction2D(self):
        mesh = UnitSquareMesh(16, 16)
        mf = FacetFunction("size_t", mesh, 12)
        file = File("facet_mf2D.x3d")
        #self.assertRaises(RuntimeError, file << mf)

    def test_save_cell_meshfunctio22D(self):
        mesh = UnitCubeMesh(16, 16, 16)
        mf = CellFunction("size_t", mesh, 12)
        file = File("cell_mf3D.x3d")
        file << mf

    def test_save_facet_meshfunction3D(self):
        mesh = UnitCubeMesh(16, 16, 16)
        mf = FacetFunction("size_t", mesh, 12)
        file = File("facet_mf3D.x3d")
        #self.assertRaises(RuntimeError, file << mf)

if __name__ == "__main__":
    unittest.main()
