"""Unit tests for local assembly"""

# Copyright (C) 2015 Tormod Landet
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import numpy
from dolfin import (UnitIntervalMesh, UnitSquareMesh,
                    Constant, Cell, TestFunction, TrialFunction, MPI, Cells, dx,
                    ds, dS, dot, Form, FunctionSpace, VectorFunctionSpace,
                    Expression, FacetNormal)
from dolfin.fem.assembling import assemble_local


@pytest.mark.skip
def test_local_assembler_1D():
    mesh = UnitIntervalMesh(MPI.comm_world, 20)
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    c = Cell(mesh, 0)

    a_scalar = Constant(1) * dx(domain=mesh)
    a_vector = v * dx
    a_matrix = u * v * dx

    A_scalar = assemble_local(a_scalar, c)
    A_vector = assemble_local(a_vector, c)
    A_matrix = assemble_local(a_matrix, c)

    assert isinstance(A_scalar, float)
    assert numpy.isclose(A_scalar, 0.05)

    assert isinstance(A_vector, numpy.ndarray)
    assert A_vector.shape == (2,)
    assert numpy.isclose(A_vector[0], 0.025)
    assert numpy.isclose(A_vector[1], 0.025)

    assert isinstance(A_matrix, numpy.ndarray)
    assert A_matrix.shape == (2, 2)
    assert numpy.isclose(A_matrix[0, 0], 1 / 60)
    assert numpy.isclose(A_matrix[0, 1], 1 / 120)
    assert numpy.isclose(A_matrix[1, 0], 1 / 120)
    assert numpy.isclose(A_matrix[1, 1], 1 / 60)


@pytest.mark.skip
def test_local_assembler_on_facet_integrals():
    mesh = UnitSquareMesh(MPI.comm_world, 4, 4, 'right')
    Vdg = FunctionSpace(mesh, 'DG', 0)
    Vdgt = FunctionSpace(mesh, 'DGT', 1)

    v = TestFunction(Vdgt)

    # Initialize DG function "w" in discontinuous pattern
    w = Expression('(1.0 + pow(x[0], 2.2) + 1/(0.1 + pow(x[1], 3)))*300.0',
                   element=Vdg.ufl_element())

    # Define form that tests that the correct + and - values are used
    L = w('-') * v('+') * dS

    # Compile form. This is collective
    L = Form(L)

    # Get global cell 10. This will return a cell only on one of the
    # processes
    c = get_cell_at(mesh, 5 / 12, 1 / 3, 0)

    if c:
        # Assemble locally on the selected cell
        b_e = assemble_local(L, c)

        # Compare to values from phonyx (fully independent
        # implementation)
        b_phonyx = numpy.array([266.55210302, 266.55210302, 365.49000122, 365.49000122, 0.0, 0.0])
        error = sum((b_e - b_phonyx)**2)**0.5
        error = float(error)  # MPI.max does strange things to numpy.float64

    else:
        error = 0.0

    error = MPI.max(MPI.comm_world, float(error))
    assert error < 1e-8


@pytest.mark.skip
def test_local_assembler_on_facet_integrals2():
    mesh = UnitSquareMesh(MPI.comm_world, 4, 4)
    Vu = VectorFunctionSpace(mesh, 'DG', 1)
    Vv = FunctionSpace(mesh, 'DGT', 1)
    u = TrialFunction(Vu)
    v = TestFunction(Vv)
    n = FacetNormal(mesh)

    # Define form
    a = dot(u, n) * v * ds
    for R in '+-':
        a += dot(u(R), n(R)) * v(R) * dS

    # Compile form. This is collective
    a = Form(a)

    # Get global cell 0. This will return a cell only on one of the
    # processes
    c = get_cell_at(mesh, 1 / 6, 1 / 12, 0)

    if c:
        A_e = assemble_local(a, c)
        A_correct = numpy.array([[0, 1 / 12, 1 / 24, 0, 0, 0],
                                 [0, 1 / 24, 1 / 12, 0, 0, 0],
                                 [-1 / 12, 0, -1 / 24, 1 / 12, 0, 1 / 24],
                                 [-1 / 24, 0, -1 / 12, 1 / 24, 0, 1 / 12],
                                 [0, 0, 0, -1 / 12, -1 / 24, 0],
                                 [0, 0, 0, -1 / 24, -1 / 12, 0]])
        error = ((A_e - A_correct)**2).sum()**0.5
        error = float(error)  # MPI.max does strange things to numpy.float64

    else:
        error = 0.0

    error = MPI.max(MPI.comm_world, float(error))
    assert error < 1e-16


def get_cell_at(mesh, x, y, z, eps=1e-3):
    """Return the cell with the given midpoint or None if not found. The
    function also checks that the cell is found on one of the
    processes when running in parallel to avoid that the above tests
    always suceed if the cell is not found on any of the processes.

    """
    found = None
    for cell in Cells(mesh):
        mp = cell.midpoint().array()
        if abs(mp[0] - x) + abs(mp[1] - y) + abs(mp[2] - z) < eps:
            found = cell
            break

    # Make sure this cell is on at least one of the parallel processes
    marker = 1 if found is not None else 0
    assert MPI.max(MPI.comm_world, marker) == 1

    return found
