# Copyright (C) 2017 Michal Habera
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

import pytest
import numpy as np
from dolfin import *
import os
from dolfin_utils.test import skip_in_parallel, fixture, tempdir

@skip_in_parallel
def test_save_2d_scalar(tempdir):

    mesh = UnitSquareMesh(16, 16)
    name = "scalar2D"
    file = File(os.path.join(tempdir, name + ".xyz"))
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = Function(V)
    u.interpolate(Constant(1.0))
    file << u
    
    # Load saved file as np array
    loaded_file = np.loadtxt(os.path.join(tempdir, name + "000000.xyz"))
    # Check if the loaded function is everywhere == 1.
    assert (loaded_file[:, 2] == 1.).all()
