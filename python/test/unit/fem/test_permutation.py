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
                    VectorElement, VectorFunctionSpace, VertexRange, cpp, fem,
                    function, interpolate, triangle)
from dolfin.cpp.mesh import GhostMode
from dolfin_utils.test.skips import skip_in_parallel


@skip_in_parallel
def test_p4_scalar_vector():

    perms = itertools.permutations([1, 2, 3, 4])

    for p in perms:
        cells = numpy.array([[0, 1, 2, 3], p], dtype=numpy.int64)
        points = numpy.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=numpy.float64)

        mesh = Mesh(MPI.comm_world, CellType.Type.tetrahedron, points, cells,
                    [], GhostMode.none)
        mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)
        Q = FunctionSpace(mesh, ("CG", 4))

        @function.expression.numba_eval
        def x0(values, x, cell_idx):
            values[:, 0] = x[:, 0]

        @function.expression.numba_eval
        def x1(values, x, cell_idx):
            values[:, 0] = x[:, 1]

        @function.expression.numba_eval
        def x2(values, x, cell_idx):
            values[:, 0] = x[:, 2]

        F0 = interpolate(Expression(x0), Q)
        F1 = interpolate(Expression(x1), Q)
        F2 = interpolate(Expression(x2), Q)

        tree = cpp.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
        pts = numpy.array([[0.4, 0.4, 0.1], [0.4, 0.1, 0.4], [0.1, 0.4, 0.4]])
        for pt in pts:
            assert numpy.isclose(pt[0], F0(pt, tree)[0])
            assert numpy.isclose(pt[1], F1(pt, tree)[0])
            assert numpy.isclose(pt[2], F2(pt, tree)[0])

        V = VectorFunctionSpace(mesh, ("CG", 4))

        @function.expression.numba_eval
        def x0x10(values, x, cell_idx):
            values[:, 0] = x[:, 0]
            values[:, 1] = x[:, 1]
            values[:, 2] = 0.0

        F = interpolate(Expression(x0x10, shape=(3,)), V)
        tree = cpp.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
        for pt in pts:
            result = F(pt, tree)
            assert numpy.isclose(pt[0], result[0])
            assert numpy.isclose(pt[1], result[1])
            assert numpy.isclose(0.0, result[2])


def test_p4_parallel_2d():
    mesh = UnitSquareMesh(MPI.comm_world, 5, 8)
    Q = FunctionSpace(mesh, ("CG", 4))
    F = Function(Q)

    @function.expression.numba_eval
    def x0(values, x, cell_idx):
        values[:, 0] = x[:, 0]

    F.interpolate(Expression(x0))

    # Generate random points in this mesh partition (one per cell)
    x = numpy.zeros(3)
    tree = cpp.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    for c in Cells(mesh):
        x[0] = random()
        x[1] = random() * (1 - x[0])
        x[2] = 1 - x[0] - x[1]
        p = Point(0.0, 0.0)
        for i, v in enumerate(VertexRange(c)):
            p += v.point() * x[i]
        p = p.array()[:2]

        assert numpy.isclose(F(p, tree)[0], p[0])


def test_p4_parallel_3d():
    mesh = UnitCubeMesh(MPI.comm_world, 3, 5, 8)
    Q = FunctionSpace(mesh, ("CG", 5))
    F = Function(Q)

    @function.expression.numba_eval
    def x0(values, x, cell_idx):
        values[:, 0] = x[:, 0]

    F.interpolate(Expression(x0))

    # Generate random points in this mesh partition (one per cell)
    x = numpy.zeros(4)
    tree = cpp.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    for c in Cells(mesh):
        x[0] = random()
        x[1] = random() * (1 - x[0])
        x[2] = random() * (1 - x[0] - x[1])
        x[3] = 1 - x[0] - x[1] - x[2]
        p = Point(0.0, 0.0, 0.0)
        for i, v in enumerate(VertexRange(c)):
            p += v.point() * x[i]
        p = p.array()

        assert numpy.isclose(F(p, tree)[0], p[0])


def test_mixed_parallel():
    mesh = UnitSquareMesh(MPI.comm_world, 5, 8)
    V = VectorElement("Lagrange", triangle, 4)
    Q = FiniteElement("Lagrange", triangle, 5)
    W = FunctionSpace(mesh, Q * V)
    F = Function(W)

    tree = cpp.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)

    @function.expression.numba_eval
    def expr_eval(values, x, cell_idx):
        values[:, 0] = x[:, 0]
        values[:, 1] = x[:, 1]
        values[:, 2] = numpy.sin(x[:, 0] + x[:, 1])

    F.interpolate(Expression(expr_eval, shape=(3,)))

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

        val = F(p, tree)
        assert numpy.allclose(val[0], p[0])
        assert numpy.isclose(val[1], p[1])
        assert numpy.isclose(val[2], numpy.sin(p[0] + p[1]))
