#!/usr/bin/env py.test

"""Unit tests for the Vector interface"""

# Copyright (C) 2011-2014 Garth N. Wells
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
#
# Modified by Anders Logg 2011

from __future__ import print_function
import pytest
from dolfin import *

from dolfin_utils.test import *


# TODO: Use the fixture setup from matrix in a shared conftest.py when
#       we move tests to one flat folder.

# Lists of backends supporting or not supporting GenericVector::data()
# access
data_backends = []
no_data_backends = ["PETSc", "Tpetra"]

# Add serial only backends
if MPI.size(mpi_comm_world()) == 1:
    # TODO: What about "Dense" and "Sparse"? The sub_backend wasn't
    # used in the old test.
    data_backends += ["Eigen"]
    no_data_backends += ["PETScCusp"]

# If we have PETSc, STL Vector gets typedefed to one of these and data
# test will not work. If none of these backends are available
# STLVector defaults to EigenVEctor, which data will work
if has_linear_algebra_backend("PETSc"):
    no_data_backends += ["STL"]
else:
    data_backends += ["STL"]

# Remove backends we haven't built with
data_backends = list(filter(has_linear_algebra_backend, data_backends))
no_data_backends = list(filter(has_linear_algebra_backend, no_data_backends))
any_backends = data_backends + no_data_backends

# Fixtures setting up and resetting the global linear algebra backend
# for a list of backends
data_backend = set_parameters_fixture("linear_algebra_backend", data_backends)
no_data_backend = set_parameters_fixture("linear_algebra_backend",
                                         no_data_backends)
any_backend = set_parameters_fixture("linear_algebra_backend", any_backends)

class TestVectorForAnyBackend:

    @pytest.fixture(autouse=True)
    def assemble_vectors(self):
        mesh = UnitSquareMesh(7, 4)

        V = FunctionSpace(mesh, "Lagrange", 2)
        W = FunctionSpace(mesh, "Lagrange", 1)

        v = TestFunction(V)
        t = TestFunction(W)

        return assemble(v*dx), assemble(t*dx)

    def test_create_empty_vector(self, any_backend):
        v0 = Vector()
        info(v0)
        info(v0, True)
        assert v0.size() == 0

    def test_create_vector(self, any_backend):
        n = 301
        v1 = Vector(mpi_comm_world(), n)
        assert v1.size() == n

    def test_copy_vector(self, any_backend):
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        v1 = Vector(v0)
        assert v0.size() == n
        del v0
        assert v1.size() == n

    def test_assign_and_copy_vector(self, any_backend):
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        v0[:] = 1.0
        assert v0.sum() == n
        v1 = Vector(v0)
        del v0
        assert v1.sum() == n

    def test_zero(self, any_backend):
        v0 = Vector(mpi_comm_world(), 301)
        v0.zero()
        assert v0.sum() == 0.0

    def test_apply(self, any_backend):
        v0 = Vector(mpi_comm_world(), 301)
        v0.apply("insert")
        v0.apply("add")

    def test_str(self, any_backend):
        v0 = Vector(mpi_comm_world(), 13)
        tmp = v0.str(False)
        tmp = v0.str(True)

    def test_init_range(self, any_backend):
        n = 301
        local_range = MPI.local_range(mpi_comm_world(), n)
        v0 = Vector()
        v0.init(mpi_comm_world(), local_range)
        assert v0.local_range() == local_range

    def test_size(self, any_backend):
        n = 301
        v0 = Vector(mpi_comm_world(), 301)
        assert v0.size() == n

    def test_local_size(self, any_backend):
        n = 301
        local_range = MPI.local_range(mpi_comm_world(), n)
        v0 = Vector()
        v0.init(mpi_comm_world(), local_range)
        assert v0.local_size() == local_range[1] - local_range[0]

    def test_owns_index(self, any_backend):
        m, n = 301, 25
        v0 = Vector(mpi_comm_world(), m)
        local_range = v0.local_range()
        in_range = local_range[0] <= n < local_range[1]
        assert v0.owns_index(n) == in_range

    #def test_set(self, any_backend):

    #def test_add(self, any_backend):

    def test_get_local(self, any_backend):
        from numpy import empty
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        data = v0.get_local()

    def test_set_local(self, any_backend):
        from numpy import zeros
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        data = zeros((v0.local_size()), dtype='d')
        v0.set_local(data)
        data = zeros((v0.local_size()*2), dtype='d')

    def test_add_local(self, any_backend):
        from numpy import zeros
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        data = zeros((v0.local_size()), dtype='d')
        v0.add_local(data)
        data = zeros((v0.local_size()*2), dtype='d')
        with pytest.raises(TypeError):
            v0.add_local(data[::2])

    #def test_gather(self, any_backend):

    def test_axpy(self, any_backend):
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        v0[:] = 1.0
        v1 = Vector(v0)
        v0.axpy(2.0, v1)
        assert v0.sum() == 2*n + n

    def test_abs(self, any_backend):
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        v0[:] = -1.0
        v0.abs()
        assert v0.sum() == n

    def test_inner(self, any_backend):
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        v0[:] = 2.0
        v1 = Vector(mpi_comm_world(), n)
        v1[:] = 3.0
        assert v0.inner(v1) == 6*n

    def test_norm(self, any_backend):
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        v0[:] = -2.0
        assert v0.norm("l1") == 2.0*n
        assert v0.norm("l2") == sqrt(4.0*n)
        assert v0.norm("linf") == 2.0

    def test_min(self, any_backend):
        v0 = Vector(mpi_comm_world(), 301)
        v0[:] = 2.0
        assert v0.min() == 2.0

    def test_max(self, any_backend):
        v0 = Vector(mpi_comm_world(),301)
        v0[:] = -2.0
        assert v0.max() == -2.0

    def test_sum(self, any_backend):
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        v0[:] = -2.0
        assert v0.sum() == -2.0*n

    def test_sum_entries(self, any_backend):
        from numpy import zeros
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        v0[:] = -2.0
        entries = zeros(5, dtype='uintp')
        assert v0.sum(entries) == -2.0
        entries[0] = 2
        entries[1] = 1
        entries[2] = 236
        entries[3] = 123
        entries[4] = 97
        assert v0.sum(entries) == -2.0*5

    def test_scalar_mult(self, any_backend):
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        v0[:] = -1.0
        v0 *= 2.0
        assert v0.sum() == -2.0*n

    def test_vector_element_mult(self, any_backend):
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        v1 = Vector(mpi_comm_world(), n)
        v0[:] = -2.0
        v1[:] =  3.0
        v0 *= v1
        assert v0.sum() == -6.0*n

    def test_scalar_divide(self, any_backend):
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        v0[:] = -1.0
        v0 /= -2.0
        assert v0.sum() == 0.5*n

    def test_vector_add(self, any_backend):
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        v1 = Vector(mpi_comm_world(), n)
        v0[:] = -1.0
        v1[:] =  2.0
        v0 += v1
        assert v0.sum() == n

    def test_scalar_add(self, any_backend):
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        v1 = Vector(mpi_comm_world(), n)
        v0[:] = -1.0
        v0 += 2.0
        assert v0.sum() == n
        v0 -= 2.0
        assert v0.sum() == -n
        v0 = v0 + 3.0
        assert v0.sum() == 2*n
        v0 = v0 - 1.0
        assert v0.sum() == n

    def test_vector_subtract(self, any_backend):
        n = 301
        v0 = Vector(mpi_comm_world(), n)
        v1 = Vector(mpi_comm_world(), n)
        v0[:] = -1.0
        v1[:] =  2.0
        v0 -= v1
        assert v0.sum() == -3.0*n

    def test_vector_assignment(self, any_backend):
        m, n = 301, 345
        v0 = Vector(mpi_comm_world(), m)
        v1 = Vector(mpi_comm_world(), n)
        v0[:] = -1.0
        v1[:] =  2.0
        v0 = v1
        assert v0.sum() == 2.0*n

    def test_vector_assignment_length(self, any_backend):
        # Test that assigning vectors of different lengths fails
        m, n = 301, 345
        v0 = Vector(mpi_comm_world(), m)
        v1 = Vector(mpi_comm_world(), n)
        def wrong_assignment(v0, v1):
            v0[:] = v1
        with pytest.raises(RuntimeError):
            wrong_assignment(v0, v1)

    def test_vector_assignment_length(self, any_backend):
        # Test that assigning with diffrent parallel layouts fails
        if MPI.size(mpi_comm_world()) > 1:
            m = 301
            local_range0 = MPI.local_range(mpi_comm_world(), m)
            print("local range", local_range0[0], local_range0[1])

            # Shift parallel partitiong but preserve global size
            if MPI.rank(mpi_comm_world()) == 0:
                local_range1 = (local_range0[0], local_range0[1] + 1)
            elif MPI.rank(mpi_comm_world()) == MPI.size(mpi_comm_world()) - 1:
                local_range1 = (local_range0[0] + 1, local_range0[1])
            else:
                local_range1 = (local_range0[0] + 1, local_range0[1] + 1)

            v0 = Vector()
            v0.init(mpi_comm_world(), local_range0)
            v1 = Vector()
            v1.init(mpi_comm_world(), local_range1)
            assert v0.size() == v1.size()

            def wrong_assignment(v0, v1):
                v0[:] = v1
                with pytest.raises(RuntimeError):
                    wrong_assignment(v0, v1)


    # Test the access of the raw data through pointers
    # This is only available for Eigen backend
    def test_vector_data(self, data_backend):
        # Test for ordinary Vector
        v = Vector(mpi_comm_world(), 301)
        v = as_backend_type(v)
        array = v.array()
        data = v.data()
        assert (data == array).all()

        # Test none writeable of a shallow copy of the data
        data = v.data(False)
        def write_data(data):
            data[0] = 1
        with pytest.raises(Exception):
            write_data(data)

        # Test for as_backend_typeed Vector
        v = as_backend_type(v)
        data = v.data()
        assert (data==array).all()
