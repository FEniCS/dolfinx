"Unit tests for XML input/output of Mesh (class XMLMesh)"

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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg 2011
#
# First added:  2011-06-17
# Last changed: 2011-09-02

import unittest
from dolfin import *

class XMLMesh(unittest.TestCase):

    def test_save_plain_mesh2D(self):
        if MPI.num_processes() == 1:
            mesh = UnitSquare(8, 8)
            f = File("unit_square.xml")
            f << mesh

    def test_save_plain_mesh3D(self):
        if MPI.num_processes() == 1:
            mesh = UnitCube(8, 8, 8)
            f = File("unit_cube.xml")
            f << mesh

    def test_vmtk_io(self):
        "Test input/output for VMTK data"

        # Read mesh with boundary indicators
        mesh = Mesh()
        f = File("../../../../data/meshes/aneurysm.xml.gz")
        f >> mesh

        # Check for generated exterior_facet_domains
        mf = mesh.data().mesh_function("exterior_facet_domains")
        self.assertEqual(mf.size(), 59912)

        # Write mesh with boundary indicators
        g = File("XMLMesh_test_vmtk_io.xml")
        g << mesh

if __name__ == "__main__":
    unittest.main()
