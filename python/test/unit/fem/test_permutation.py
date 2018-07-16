"""Unit tests for the fem interface"""

# Copyright (C) 2009-2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from dolfin import (Mesh, MPI, CellType, fem, FunctionSpace,
                    VectorFunctionSpace, interpolate, Expression)
from dolfin.cpp.mesh import GhostMode
from dolfin_utils.test import skip_in_parallel
import numpy
import itertools

@skip_in_parallel
def test_p4_scalar_vector():

    perms = itertools.permutations([1, 2, 3, 4])

    for p in perms:
        print(p)
        cells = numpy.array([[0, 1, 2, 3], p], dtype=numpy.int64)
        points = numpy.array([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0],
                              [1.0, 1.0, 1.0]], dtype=numpy.float64)

        mesh = Mesh(MPI.comm_world, CellType.Type.tetrahedron, points, cells, [], GhostMode.none)
        mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)

        Q = FunctionSpace(mesh, "CG", 4)
        F0 = interpolate(Expression("x[0]", degree=4), Q)
        F1 = interpolate(Expression("x[1]", degree=4), Q)
        F2 = interpolate(Expression("x[2]", degree=4), Q)

        pts = numpy.array([[0.4, 0.4, 0.1], [0.4, 0.1, 0.4], [0.1, 0.4, 0.4]])

        for pt in pts:
            print(pt, F0(pt), F1(pt), F2(pt))
            assert numpy.isclose(pt[0], F0(pt)[0])
            assert numpy.isclose(pt[1], F1(pt)[0])
            assert numpy.isclose(pt[2], F2(pt)[0])

        V = VectorFunctionSpace(mesh, "CG", 4)
        F = interpolate(Expression(("x[0]", "x[1]", "0.0"), degree=4), V)
        for pt in pts:
            result = F(pt)[0]
            print(pt, result)
            assert numpy.isclose(pt[0], result[0])
            assert numpy.isclose(pt[1], result[1])
            assert numpy.isclose(0.0, result[2])
