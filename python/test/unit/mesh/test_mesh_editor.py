# Copyright (C) 2006-2011 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import Mesh, MPI, CellType, cpp
from dolfin_utils.test import skip_in_parallel
import numpy


@skip_in_parallel
def test_triangle_mesh():

    # Create mesh object
    mesh = Mesh(MPI.comm_world, CellType.Type.triangle,
                numpy.array([[0.0, 0.0],
                             [1.0, 0.0],
                            [0.0, 1.0]], dtype=numpy.float64),
                numpy.array([[0, 1, 2]], dtype=numpy.int32), [],
                cpp.mesh.GhostMode.none)
    assert mesh.num_entities_global(0) == 3
    assert mesh.num_entities_global(2) == 1
