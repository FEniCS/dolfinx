# Copyright (C) 2006-2011 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy

from dolfin import MPI, CellType, Mesh, cpp
from dolfin_utils.test.skips import skip_in_parallel


@skip_in_parallel
def test_triangle_mesh():
    mesh = Mesh(MPI.comm_world, CellType.triangle,
                numpy.array([[0.0, 0.0],
                             [1.0, 0.0],
                            [0.0, 1.0]], dtype=numpy.float64),
                numpy.array([[0, 1, 2]], dtype=numpy.int32), [],
                cpp.mesh.GhostMode.none)
    assert mesh.num_entities_global(0) == 3
    assert mesh.num_entities_global(2) == 1
