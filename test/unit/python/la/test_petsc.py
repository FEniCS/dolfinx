#!/usr/bin/env py.test

"""Unit tests for parts of the PETSc interface not tested via the
GenericFoo interface"""

# Copyright (C) 2015 Garth N. Wells
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

from __future__ import print_function
import pytest
from dolfin import *
from dolfin_utils.test import skip_if_not_PETSc, skip_in_parallel


@skip_if_not_PETSc
def test_options_prefix():
    "Test set/get prefix option for PETSc objects"

    def run_test(A, init_function):
        # Prefix
        prefix = "test_foo_"

        # Set prefix
        A.set_options_prefix(prefix)

        # Get prefix (should be empty since vector has been initialised)
        assert not A.get_options_prefix()

        # Initialise vector
        init_function(A)

        # Check prefix
        assert A.get_options_prefix() == prefix

        # Try changing prefix post-intialisation (should throw error)
        with pytest.raises(RuntimeError):
            A.set_options_prefix("test")

    # Test vector
    def init_vector(x):
        x.init(mpi_comm_world(), 100)
    x = PETScVector()
    run_test(x, init_vector)

    # Test matrix
    def init_matrix(A):
        mesh = UnitSquareMesh(12, 12)
        V = FunctionSpace(mesh, "Lagrange", 1)
        u, v = TrialFunction(V), TestFunction(V)
        assemble(u*v*dx, tensor=A)
    A = PETScMatrix()
    run_test(A, init_matrix)

    # FIXME: Need to straighten out the calls to PETSc
    # FooSetFromOptions calls in the solver wrapers

    # Test solvers
    #def init_solver(A, solver):
    #    A = PETScMatrix()
    #    init_matrix(A)
    #    solver.set_operator(A)

    # Test Krylov solver
    #solver = PETScKrylovSolver()
    #run_test(solver, init_solver)

    # Test KY solver
    #solver = PETScLUSolver()
    #run_test(solver, init_solver)
