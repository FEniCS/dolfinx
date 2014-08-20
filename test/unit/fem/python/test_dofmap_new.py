"""Unit tests for the DofMap interface"""

# Copyright (C) 2014 Garth N. Wells
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

from __future__ import print_function
import pytest
import numpy as np
from dolfin import *

def test_dofmap_clear_submap():
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "Lagrange", 1)
    W = V*V

    # Check block size
    assert W.dofmap().block_size == 2

    W.dofmap().clear_sub_map_data()
    with pytest.raises(RuntimeError):
        W0 = W.sub(0)
    with pytest.raises(RuntimeError):
        W1 = W.sub(1)
