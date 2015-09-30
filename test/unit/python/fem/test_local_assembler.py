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
    assert A_matrix.shape == (2,2)
    assert near(A_matrix[0,0], 1/60)
    assert near(A_matrix[0,1], 1/120)
    assert near(A_matrix[1,0], 1/120)
    assert near(A_matrix[1,1], 1/60)


def test_local_assembler_on_facet_integrals():
    mesh = UnitSquareMesh(4, 4, 'right')
    Vdg = VectorFunctionSpace(mesh, 'CG', 1)
    Vdgt = FunctionSpace(mesh, 'CG', 1) 
    
    v = TestFunction(Vdgt)
    n = FacetNormal(mesh) 
    
    w = Function(Vdg)
    for cell in cells(mesh):
        for dof in Vdg.dofmap().cell_dofs(cell.index()):
            w.vector()[dof] = 1.0 + cell.index() % 5
    
    L = dot(w('-'), w('+'))*v('+')*dS
    c = Cell(mesh, 5)
    b_e = assemble_local(L, c)
    b_a = numpy.array([0, 19/8, 9/8])
    error = sum((b_e - b_a)**2)
    assert error < 1e-16


def test_local_assembler_on_facet_integrals2():
    mesh = UnitSquareMesh(4, 4)
    Vu = VectorFunctionSpace(mesh, 'DG', 1)
    Vv = FunctionSpace(mesh, 'DGT', 1)
    u = TrialFunction(Vu)
    v = TestFunction(Vv)
    n = FacetNormal(mesh)

    a = dot(u, n)*v*ds
    for R in '+-':
        a += dot(u(R), n(R))*v(R)*dS

    c = Cell(mesh, 0)
    A_e = assemble_local(a, c)
    A_correct = numpy.array([[    0, 1/12,  1/24,     0,     0,    0],
                             [    0, 1/24,  1/12,     0,     0,    0],
                             [-1/12,    0, -1/24,  1/12,     0, 1/24],
                             [-1/24,    0, -1/12,  1/24,     0, 1/12],
                             [    0,    0,     0, -1/12, -1/24,    0],
                             [    0,    0,     0, -1/24, -1/12,    0]])
    error = ((A_e - A_correct)**2).sum()**0.5
    assert error < 1e-15
