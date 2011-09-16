"Unit tests for XML input/output of MeshValueCollection"

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
# First added:  2011-09-01
# Last changed: 2011-09-01

import unittest
from dolfin import *

class XMLMeshValueCollection(unittest.TestCase):

    def test_io(self):
        "Test input/output"

        # Create mesh value collection and add some data
        mesh = UnitCube(5, 5, 5)
        output_values = MeshValueCollection("uint", 2)
        output_values.set_value(1, 1, 1);
        output_values.set_value(2, 1, 3);
        output_values.set_value(5, 1, 8);
        output_values.set_value(13, 1, 21);

        # Write to file
        output_file = File("XMLMeshValueCollection_test_io.xml")
        output_file << output_values

        # Read from file
        input_file = File("XMLMeshValueCollection_test_io.xml")
        input_values = MeshValueCollection("uint", 2)
        input_file >> input_values

        # Get some data and check that it matches
        self.assertEqual(input_values.size(), output_values.size())

if __name__ == "__main__":
    unittest.main()
