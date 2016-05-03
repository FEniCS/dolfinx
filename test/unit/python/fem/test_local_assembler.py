#!/usr/bin/env py.test

"""Unit tests for local assembly"""

# Copyright (C) 2015 Tormod Landet
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

from __future__ import division
import numpy
from dolfin import *
from dolfin_utils.test import set_parameters_fixture


ghost_mode = set_parameters_fixture("ghost_mode", ["shared_facet"])


def test_local_assembler_1D():
    mesh = UnitIntervalMesh(20)
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    c = Cell(mesh, 0)

    a_scalar = Constant(1)*dx(domain=mesh)
    a_vector = v*dx
    a_matrix = u*v*dx

    A_scalar = assemble_local(a_scalar, c)
    A_vector = assemble_local(a_vector, c)
    A_matrix = assemble_local(a_matrix, c)

    assert isinstance(A_scalar, float)
    assert near(A_scalar, 0.05)

    assert isinstance(A_vector, numpy.ndarray)
    assert A_vector.shape == (2,)
    assert near(A_vector[0], 0.025)
    assert near(A_vector[1], 0.025)

    assert isinstance(A_matrix, numpy.ndarray)
    assert A_matrix.shape == (2, 2)
    assert near(A_matrix[0, 0], 1/60)
    assert near(A_matrix[0, 1], 1/120)
    assert near(A_matrix[1, 0], 1/120)
    assert near(A_matrix[1, 1], 1/60)


def test_local_assembler_on_facet_integrals(ghost_mode):
    mesh = UnitSquareMesh(4, 4, 'right')
    Vcg = FunctionSpace(mesh, 'CG', 1)
    Vdg = FunctionSpace(mesh, 'DG', 0)
    Vdgt = FunctionSpace(mesh, 'DGT', 1)

    v = TestFunction(Vdgt)
    n = FacetNormal(mesh)

    # Initialize DG function "w" in discontinuous pattern
    w = Expression('(1.0 + pow(x[0], 2.2) + 1/(0.1 + pow(x[1], 3)))*300.0',
                   element=Vdg.ufl_element())

    # Define form that tests that the correct + and - values are used
    L = w('-')*v('+')*dS

    # Compile form. This is collective
    L = Form(L)

    # Get global cell 10. This will return a cell only on one of the
    # processes
    c = get_cell_at(mesh, 5/12, 1/3, 0)

    if c:
        # Assemble locally on the selected cell
        b_e = assemble_local(L, c)

        # Compare to values from phonyx (fully independent
        # implementation)
        b_phonyx = numpy.array([266.55210302, 266.55210302, 365.49000122,
                                365.49000122,          0.0,          0.0])
        error = sum((b_e - b_phonyx)**2)**0.5
        error = float(error)  # MPI.max does strange things to numpy.float64

    else:
        error = 0.0

    error = MPI.max(mpi_comm_world(), float(error))
    assert error < 1e-8


def test_local_assembler_on_facet_integrals2(ghost_mode):
    mesh = UnitSquareMesh(4, 4)
    Vu = VectorFunctionSpace(mesh, 'DG', 1)
    Vv = FunctionSpace(mesh, 'DGT', 1)
    u = TrialFunction(Vu)
    v = TestFunction(Vv)
    n = FacetNormal(mesh)

    # Define form
    a = dot(u, n)*v*ds
    for R in '+-':
        a += dot(u(R), n(R))*v(R)*dS

    # Compile form. This is collective
    a = Form(a)

    # Get global cell 0. This will return a cell only on one of the
    # processes
    c = get_cell_at(mesh, 1/6, 1/12, 0)

    if c:
        A_e = assemble_local(a, c)
        A_correct = numpy.array([[0, 1/12, 1/24, 0, 0, 0],
                                 [0, 1/24, 1/12, 0, 0, 0],
                                 [-1/12, 0, -1/24, 1/12, 0, 1/24],
                                 [-1/24, 0, -1/12, 1/24, 0, 1/12],
                                 [0, 0, 0, -1/12, -1/24, 0],
                                 [0, 0, 0, -1/24, -1/12, 0]])
        error = ((A_e - A_correct)**2).sum()**0.5
        error = float(error)  # MPI.max does strange things to numpy.float64

    else:
        error = 0.0

    error = MPI.max(mpi_comm_world(), float(error))
    assert error < 1e-16


def get_cell_at(mesh, x, y, z, eps=1e-3):
    """Return the cell with the given midpoint or None if not found. The
    function also checks that the cell is found on one of the
    processes when running in parallel to avoid that the above tests
    always suceed if the cell is not found on any of the processes.

    """
    found = None
    for cell in cells(mesh):
        mp = cell.midpoint()
        if abs(mp.x() - x) + abs(mp.y() - y) + abs(mp.y() - y) < eps:
            found = cell
            break

    # Make sure this cell is on at least one of the parallel processes
    marker = 1 if found is not None else 0
    assert MPI.max(mpi_comm_world(), marker) == 1

    return found
