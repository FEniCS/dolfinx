"Unit tests for input/output of mesh data"

# Copyright (C) 2011 Anders Logg
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
# First added:  2011-08-22
# Last changed: 2011-08-25

import unittest
from dolfin import *

class XMLMeshData(unittest.TestCase):

    def test_io(self):
        "Test input/output"

        # Read mesh with boundary indicators
        mesh = Mesh()
        f = File("../../../../data/meshes/aneurysm.xml.gz")
        f >> mesh

        # Check for generated exterior_facet_domains
        mf = mesh.data().mesh_function("exterior_facet_domains")
        self.assertEqual(mf.size(), 59912)

        # Write mesh with boundary indicators
        g = File("MeshData_test_io.xml")
        g << mesh

if __name__ == "__main__":
    unittest.main()
