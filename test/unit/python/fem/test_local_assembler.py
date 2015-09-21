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

import numpy
from dolfin import *

def test_local_assembler_1D():
    mesh = UnitIntervalMesh(2)
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
    assert A_scalar == 0.5
    
    assert isinstance(A_vector, numpy.ndarray)
    assert A_vector.shape == (2,)
    assert near(A_vector[0], 0.25)
    assert near(A_vector[1], 0.25) 
    
    assert isinstance(A_matrix, numpy.ndarray)
    assert A_matrix.shape == (2,2)
    assert near(A_matrix[0,0], 1/6.0)
    assert near(A_matrix[0,1], 1/12.0)
    assert near(A_matrix[1,0], 1/12.0)
    assert near(A_matrix[1,1], 1/6.0)
