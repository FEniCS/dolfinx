"""Unit tests for the Attribute interface of the HDF5 io library"""

# Copyright (C) 2013 Chris Richardson
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
#
# First added:  2013-10-25
# Last changed: 2013-10-25

import unittest
from dolfin import *

if has_hdf5():
    class HDF5_Attribute(unittest.TestCase):
        """Test read/write of Atrtibutes in HDF5 files"""

        def test_read_write_str_attribute(self):
            hdf_file = HDF5File("a.h5", "w")
            x = Vector(123)
            hdf_file.write(x, "/a_vector")
            attr = hdf_file.attributes("/a_vector")
            attr['name'] = 'Vector'
            self.assertEqual(attr.type_str("name"), "string")
            self.assertEqual(attr['name'], 'Vector')

        def test_read_write_float_attribute(self):
            hdf_file = HDF5File("a.h5", "w")
            x = Vector(123)
            hdf_file.write(x, "/a_vector")
            attr = hdf_file.attributes("/a_vector")
            attr['val'] = -9.2554            
            self.assertEqual(attr.type_str("val"), "float")
            self.assertEqual(attr['val'], -9.2554)

        def test_read_write_int_attribute(self):
            hdf_file = HDF5File("a.h5", "w")
            x = Vector(123)
            hdf_file.write(x, "/a_vector")
            attr = hdf_file.attributes("/a_vector")
            attr['val'] = 1
            self.assertEqual(attr.type_str("val"), "int")
            self.assertEqual(attr['val'], 1)

        def test_read_write_vec_float_attribute(self):
            import numpy
            hdf_file = HDF5File("a.h5", "w")
            x = Vector(123)
            hdf_file.write(x, "/a_vector")
            attr = hdf_file.attributes("/a_vector")
            vec = numpy.array([1,2,3,4.5], dtype='float')
            attr['val'] = vec
            ans = attr['val']
            self.assertEqual(attr.type_str("val"), "vectorfloat")
            self.assertEqual(len(vec), len(ans))
            for val1, val2 in zip(vec, ans):
                self.assertEqual(val1, val2)

if __name__ == "__main__":
    unittest.main()
