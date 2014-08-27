#!/usr/bin/env py.test

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

import pytest
from dolfin import *

if has_hdf5():
    def test_read_write_str_attribute():
        hdf_file = HDF5File(mpi_comm_world(), "a.h5", "w")
        x = Vector(mpi_comm_world(), 123)
        hdf_file.write(x, "/a_vector")
        attr = hdf_file.attributes("/a_vector")
        attr['name'] = 'Vector'
        assert attr.type_str("name") == "string"
        assert attr['name'] == 'Vector'

    def test_read_write_float_attribute():
        hdf_file = HDF5File(mpi_comm_world(), "a.h5", "w")
        x = Vector(mpi_comm_world(), 123)
        hdf_file.write(x, "/a_vector")
        attr = hdf_file.attributes("/a_vector")
        attr['val'] = -9.2554
        assert attr.type_str("val") == "float"
        assert attr['val'] == -9.2554

    def test_read_write_int_attribute():
        hdf_file = HDF5File(mpi_comm_world(), "a.h5", "w")
        x = Vector(mpi_comm_world(), 123)
        hdf_file.write(x, "/a_vector")
        attr = hdf_file.attributes("/a_vector")
        attr['val'] = 1
        assert attr.type_str("val") == "int"
        assert attr['val'] == 1

    def test_read_write_vec_float_attribute():
        import numpy
        hdf_file = HDF5File(mpi_comm_world(), "a.h5", "w")
        x = Vector(mpi_comm_world(), 123)
        hdf_file.write(x, "/a_vector")
        attr = hdf_file.attributes("/a_vector")
        vec = numpy.array([1,2,3,4.5], dtype='float')
        attr['val'] = vec
        ans = attr['val']
        assert attr.type_str("val") == "vectorfloat"
        assert len(vec) == len(ans)
        for val1, val2 in zip(vec, ans):
            assert val1 == val2

    def test_read_write_vec_int_attribute():
        import numpy
        hdf_file = HDF5File(mpi_comm_world(), "a.h5", "w")
        x = Vector(mpi_comm_world(), 123)
        hdf_file.write(x, "/a_vector")
        attr = hdf_file.attributes("/a_vector")
        vec = numpy.array([1,2,3,4,5], dtype=numpy.uintp)
        attr['val'] = vec
        ans = attr['val']
        assert attr.type_str("val") == "vectorint"
        assert len(vec) == len(ans)
        for val1, val2 in zip(vec, ans):
            assert val1 == val2
