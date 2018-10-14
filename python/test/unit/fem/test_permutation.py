# Copyright (C) 2009-2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the fem interface"""

import itertools
from random import random

import numpy

from dolfin import (MPI, Cells, CellType, Expression, FiniteElement, Function,
                    FunctionSpace, Mesh, Point, UnitCubeMesh, UnitSquareMesh,
                    VectorElement, VectorFunctionSpace, VertexRange, fem,
                    interpolate, triangle)
from dolfin.cpp.mesh import GhostMode
from dolfin_utils.test.skips import skip_in_parallel


@skip_in_parallel
def test_p4_scalar_vector():

    perms = itertools.permutations([1, 2, 3, 4])

    for p in perms:
        print(p)
        cells = numpy.array([[0, 1, 2, 3], p], dtype=numpy.int64)
        points = numpy.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=numpy.float64)

        mesh = Mesh(MPI.comm_world, CellType.Type.tetrahedron, points, cells,
                    [], GhostMode.none)
        mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)

        Q = FunctionSpace(mesh, ("CG", 4))
        F0 = interpolate(Expression("x[0]", degree=4), Q)
        F1 = interpolate(Expression("x[1]", degree=4), Q)
        F2 = interpolate(Expression("x[2]", degree=4), Q)

        pts = numpy.array([[0.4, 0.4, 0.1], [0.4, 0.1, 0.4], [0.1, 0.4, 0.4]])

        print("f type", type(F0))
        for pt in pts:
            print(pt, F0(pt), F1(pt), F2(pt))
            assert numpy.isclose(pt[0], F0(pt)[0])
            assert numpy.isclose(pt[1], F1(pt)[0])
            assert numpy.isclose(pt[2], F2(pt)[0])

        V = VectorFunctionSpace(mesh, ("CG", 4))
        F = interpolate(Expression(("x[0]", "x[1]", "0.0"), degree=4), V)
        for pt in pts:
            result = F(pt)
            print(pt, result)
            assert numpy.isclose(pt[0], result[0])
            assert numpy.isclose(pt[1], result[1])
            assert numpy.isclose(0.0, result[2])


def test_p4_parallel_2d():
    mesh = UnitSquareMesh(MPI.comm_world, 5, 8)
    Q = FunctionSpace(mesh, ("CG", 4))
    F = Function(Q)
    F.interpolate(Expression("x[0]", degree=4))

    # Generate random points in this mesh partition (one per cell)
    x = numpy.zeros(3)
    for c in Cells(mesh):
        x[0] = random()
        x[1] = random() * (1 - x[0])
        x[2] = 1 - x[0] - x[1]
        p = Point(0.0, 0.0)
        for i, v in enumerate(VertexRange(c)):
            p += v.point() * x[i]
        p = p.array()[:2]

        assert numpy.isclose(F(p)[0], p[0])


def test_p4_parallel_3d():
    mesh = UnitCubeMesh(MPI.comm_world, 3, 5, 8)
    Q = FunctionSpace(mesh, ("CG", 5))
    F = Function(Q)
    F.interpolate(Expression("x[0]", degree=5))

    # Generate random points in this mesh partition (one per cell)
    x = numpy.zeros(4)
    for c in Cells(mesh):
        x[0] = random()
        x[1] = random() * (1 - x[0])
        x[2] = random() * (1 - x[0] - x[1])
        x[3] = 1 - x[0] - x[1] - x[2]
        p = Point(0.0, 0.0, 0.0)
        for i, v in enumerate(VertexRange(c)):
            p += v.point() * x[i]
        p = p.array()

        assert numpy.isclose(F(p)[0], p[0])


def test_mixed_parallel():
    mesh = UnitSquareMesh(MPI.comm_world, 5, 8)
    V = VectorElement("Lagrange", triangle, 4)
    Q = FiniteElement("Lagrange", triangle, 5)
    W = FunctionSpace(mesh, Q * V)
    F = Function(W)
    F.interpolate(Expression(("x[0]", "x[1]", "sin(x[0] + x[1])"), degree=5))

    # Generate random points in this mesh partition (one per cell)
    x = numpy.zeros(3)
    for c in Cells(mesh):
        x[0] = random()
        x[1] = random() * (1 - x[0])
        x[2] = (1 - x[0] - x[1])
        p = Point(0.0, 0.0)
        for i, v in enumerate(VertexRange(c)):
            p += v.point() * x[i]
        p = p.array()[:2]

        val = F(p)
        assert numpy.allclose(val[0], p[0])
        assert numpy.isclose(val[1], p[1])
        assert numpy.isclose(val[2], numpy.sin(p[0] + p[1]))
