# Copyright (C) 2019 Jorgen Dokken, Matthew Scroggs and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import numpy as np
from mpi4py import MPI
from random import shuffle

import ufl
from dolfinx import FunctionSpace
from dolfinx.mesh import create_mesh
from dolfinx_utils.test.skips import skip_in_parallel

from pipeline_tests import run_vector_test


@skip_in_parallel
def test_N1curl_order_2_tetrahedron():
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "tetrahedron", 1))

    temp_points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.],
                            [0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]])
    order = list(range(8))
    shuffle(order)
    points = np.zeros(temp_points.shape)
    for i, j in enumerate(order):
        points[j] = temp_points[i]

    cells = []
    for cell in [[0, 1, 3, 5], [0, 3, 2, 6], [0, 4, 5, 6], [3, 5, 6, 7], [0, 3, 5, 6]]:
        shuffle(cell)
        cells.append([order[i] for i in cell])

    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    V = FunctionSpace(mesh, ("RT", 2))
    run_vector_test(mesh, V, 1)
