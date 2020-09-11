# Copyright (C) 2019-2020 Matthew Scroggs, Jorgen Dokken and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from math import log
from random import random
import numpy as np
import ufl
from dolfinx import Function, FunctionSpace
from dolfinx.fem import assemble_vector
from dolfinx.mesh import create_mesh
from dolfinx_utils.test.skips import skip_in_parallel, skip_if_complex
from mpi4py import MPI
from ufl import dx, div


@skip_in_parallel
@skip_if_complex
def test_div():
    points = np.array([[0., 0.], [0., 1.], [1., 0.], [2., 1.]])
    cells = np.array([[0, 1, 2, 3]])

    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "quadrilateral", 1))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    mesh.topology.create_connectivity_all()

    RT = FunctionSpace(mesh, ("RTCF", 1))
    tau = ufl.TestFunction(RT)
    a = div(tau) * dx
    v = assemble_vector(a)

    v = sorted(list(v[:]))

    # Assert that these values match those computed elsewhere using sympy
    actual = [-1.0, 1 / 2 - 2 * log(2), 1, 1 / 2 + log(2)]
    for a, b in zip(v, actual):
        assert abs(a - b) < 0.002


@skip_in_parallel
@skip_if_complex
def test_eval():
    points = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    cells = np.array([[0, 1, 2, 3]])

    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "quadrilateral", 1))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    mesh.topology.create_connectivity_all()

    RT = FunctionSpace(mesh, ("RTCF", 1))
    tau = Function(RT)
    basis = [lambda x: (1 - x[0], 0),
             lambda x: (x[0], 0),
             lambda x: (0, 1 - x[1]),
             lambda x: (0, x[1])]
    for i, f in zip(RT.dofmap.cell_dofs(0), basis):
        tau.vector[:] = [1 if i == j else 0 for j in range(4)]
        for count in range(5):
            point = [random(), random(), 0]
            value = tau.eval([point], [0])
            actual = f(point)
            assert np.allclose(value, actual)
