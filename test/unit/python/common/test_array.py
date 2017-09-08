"""Unit tests for Array passing"""

# Copyright (C) 2017 Tormod Landet
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


import dolfin
import numpy


def test_array_int():
    code = """
    int test_arrays(const Array<int>& int_arr)
    {
        int ret = 0;
        for (int i = 0; i < int_arr.size(); i++)
        {
            ret += int_arr[i];
        }
        return ret;
    }
    """
    module = dolfin.compile_extension_module(code=code,
                                             source_directory='.',
                                             sources=[],
                                             include_dirs=["."])
    arr = numpy.array([1, 2, 4, 8], dtype=numpy.intc)
    ans = module.test_arrays(arr)
    assert ans == arr.sum() == 15
