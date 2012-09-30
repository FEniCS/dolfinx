"""Unit tests for the XDMF io library"""

# Copyright (C) 2012 Garth N. Wells
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
# First added:  2012-09-14
# Last changed:

import unittest
from dolfin import *

class HDF5_Vector(unittest.TestCase):
    """Test input/output of Vector to HDF5 files"""

    def test_save_vector(self):
        x = Vector(305)
        x[:] = 1.0
        vector_file = File("x.h5")
        vector_file << x

    def test_save_and_read_vector(self):
        x = Vector(305)
        x[:] = 1.2
        vector_file = File("vector.h5")
        vector_file << x
        del vector_file

        y = Vector()
        vector_file = File("vector.h5")
        vector_file >> y
        self.assertEqual(y.size(), x.size())
        self.assertEqual((x - y).norm("l1"), 0.0)

if __name__ == "__main__":
    if has_hdf5():
        unittest.main()
