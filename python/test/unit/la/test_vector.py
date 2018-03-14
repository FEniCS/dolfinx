"""Unit tests for the Vector interface"""

# Copyright (C) 2011-2014 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import numpy
from copy import copy

from dolfin import *
from dolfin_utils.test import *

# TODO: Use the fixture setup from matrix in a shared conftest.py when
#       we move tests to one flat folder.

# Lists of backends supporting or not supporting GenericVector::data()
# access
data_backends = []
no_data_backends = ["PETSc", "Tpetra"]

# Add serial only backends
if MPI.size(MPI.comm_world) == 1:
    # TODO: What about "Dense" and "Sparse"? The sub_backend wasn't
    # used in the old test.
    data_backends += ["Eigen"]

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

    def test_create_empty_vector(self, any_backend):
        v0 = Vector()
        info(v0)
        info(v0, True)
        assert v0.size() == 0

    def test_create_vector(self, any_backend):
        n = 301
        v1 = Vector(MPI.comm_world, n)
        assert v1.size() == n

    def test_copy_vector(self, any_backend):
        n = 301
        v0 = Vector(MPI.comm_world, n)
        v1 = Vector(v0)
        assert v0.size() == n
        del v0
        assert v1.size() == n

    def test_assign_and_copy_vector(self, any_backend):
        n = 301
        v0 = Vector(MPI.comm_world, n)
        v0[:] = 1.0
        assert v0.sum() == n
        v1 = Vector(v0)
        del v0
        assert v1.sum() == n

    def test_zero(self, any_backend):
        v0 = Vector(MPI.comm_world, 301)
        v0.zero()
        assert v0.sum() == 0.0

    def test_apply(self, any_backend):
        v0 = Vector(MPI.comm_world, 301)
        v0.apply("insert")
        v0.apply("add")

    def test_str(self, any_backend):
        v0 = Vector(MPI.comm_world, 13)
        tmp = v0.str(False)
        tmp = v0.str(True)

    def test_init_range(self, any_backend):
        n = 301
        local_range = MPI.local_range(MPI.comm_world, n)
        v0 = Vector(MPI.comm_world)
        v0.init(local_range)
        assert v0.local_range() == local_range

    def test_size(self, any_backend):
        n = 301
        v0 = Vector(MPI.comm_world, 301)
        assert v0.size() == n

    def test_local_size(self, any_backend):
        n = 301
        local_range = MPI.local_range(MPI.comm_world, n)
        v0 = Vector(MPI.comm_world)
        v0.init(local_range)
        assert v0.local_size() == local_range[1] - local_range[0]

    def test_owns_index(self, any_backend):
        m, n = 301, 25
        v0 = Vector(MPI.comm_world, m)
        local_range = v0.local_range()
        in_range = local_range[0] <= n < local_range[1]
        assert v0.owns_index(n) == in_range

    #def test_set(self, any_backend):

    #def test_add(self, any_backend):

    def test_get_local(self, any_backend):
        from numpy import empty
        n = 301
        v0 = Vector(MPI.comm_world, n)
        data = v0.get_local()

    def test_set_local(self, any_backend):
        from numpy import zeros
        n = 301
        v0 = Vector(MPI.comm_world, n)
        data = zeros((v0.local_size()), dtype='d')
        v0.set_local(data)
        data = zeros((v0.local_size()*2), dtype='d')

    def test_add_local(self, any_backend):
        from numpy import zeros
        n = 301
        v0 = Vector(MPI.comm_world, n)
        data = zeros((v0.local_size()), dtype='d')
        v0.add_local(data)
        data = zeros((v0.local_size()*2), dtype='d')

    def test_gather(self, any_backend):
        # Gather not implemented in Eigen
        if any_backend == "Eigen" or any_backend == "Tpetra":
            return

        # Create distributed vector of local size 1
        x = DefaultFactory().create_vector(MPI.comm_world)
        r = MPI.rank(x.mpi_comm())
        x.init((r, r+1))

        # Create local vector
        y = DefaultFactory().create_vector(MPI.comm_self)

        # Do the actual test across all rank permutations
        for target_rank in range(MPI.size(x.mpi_comm())):

            # Set nonzero value on single rank
            if r == target_rank:
                x[0] = 42.0  # Set using local index
            else:
                x[0] = 0.0  # Set using local index
            assert numpy.isclose(x.sum(), 42.0)

            # Gather (using global index) and check the result
            x.gather(y, numpy.array([target_rank], dtype=la_index_dtype()))
            assert numpy.isclose(y[0], 42.0)

            # NumPy array version
            out = x.gather(numpy.array([target_rank], dtype=la_index_dtype()))
            assert out.shape == (1,) and numpy.isclose(out[0], 42.0)

            # Test gather on zero
            out = x.gather_on_zero()
            if r == 0:
                expected = numpy.array([42.0 if i == target_rank else 0.0
                                        for i in range(x.size())])
            else:
                expected = numpy.array([])
            assert out.shape == expected.shape and numpy.allclose(out, expected)

            # Test also the corner case of empty indices on one process
            if r == target_rank:
                out = x.gather(numpy.array([], dtype=la_index_dtype()))
                expected = numpy.array([])
            else:
                out = x.gather(numpy.array([target_rank], dtype=la_index_dtype()))
                expected = numpy.array([42.0])
            assert out.shape == expected.shape and numpy.allclose(out, expected)

        # Check that distributed gather vector is not accepted
        if MPI.size(MPI.comm_world) > 1:
            z = DefaultFactory().create_vector(MPI.comm_world)
            with pytest.raises(RuntimeError):
                x.gather(z, numpy.array([0], dtype=la_index_dtype()))

        # Check that gather vector of wrong size is not accepted
        z = DefaultFactory().create_vector(MPI.comm_self)
        z.init(3)
        with pytest.raises(RuntimeError):
            x.gather(z, numpy.array([0], dtype=la_index_dtype()))

    def test_axpy(self, any_backend):
        n = 301
        v0 = Vector(MPI.comm_world, n)
        v0[:] = 1.0
        v1 = Vector(v0)
        v0.axpy(2.0, v1)
        assert v0.sum() == 2*n + n

    def test_abs(self, any_backend):
        n = 301
        v0 = Vector(MPI.comm_world, n)
        v0[:] = -1.0
        v0.abs()
        assert v0.sum() == n

    def test_inner(self, any_backend):
        n = 301
        v0 = Vector(MPI.comm_world, n)
        v0[:] = 2.0
        v1 = Vector(MPI.comm_world, n)
        v1[:] = 3.0
        assert v0.inner(v1) == 6*n

    def test_norm(self, any_backend):
        n = 301
        v0 = Vector(MPI.comm_world, n)
        v0[:] = -2.0
        assert v0.norm("l1") == 2.0*n
        assert v0.norm("l2") == sqrt(4.0*n)
        assert v0.norm("linf") == 2.0

    def test_min(self, any_backend):
        v0 = Vector(MPI.comm_world, 301)
        v0[:] = 2.0
        assert v0.min() == 2.0

    def test_max(self, any_backend):
        v0 = Vector(MPI.comm_world,301)
        v0[:] = -2.0
        assert v0.max() == -2.0

    def test_sum(self, any_backend):
        n = 301
        v0 = Vector(MPI.comm_world, n)
        v0[:] = -2.0
        assert v0.sum() == -2.0*n

    def test_sum_entries(self, any_backend):
        from numpy import zeros
        n = 301
        v0 = Vector(MPI.comm_world, n)
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
        v0 = Vector(MPI.comm_world, n)
        v0[:] = -1.0
        v0 *= 2.0
        assert v0.sum() == -2.0*n

    def test_vector_element_mult(self, any_backend):
        n = 301
        v0 = Vector(MPI.comm_world, n)
        v1 = Vector(MPI.comm_world, n)
        v0[:] = -2.0
        v1[:] =  3.0
        v0 *= v1
        assert v0.sum() == -6.0*n

    def test_scalar_divide(self, any_backend):
        n = 301
        v0 = Vector(MPI.comm_world, n)
        v0[:] = -1.0
        v0 /= -2.0
        assert v0.sum() == 0.5*n

    def test_vector_add(self, any_backend):
        n = 301
        v0 = Vector(MPI.comm_world, n)
        v1 = Vector(MPI.comm_world, n)
        v0[:] = -1.0
        v1[:] =  2.0
        v0 += v1
        assert v0.sum() == n

    def test_scalar_add(self, any_backend):
        n = 301
        v0 = Vector(MPI.comm_world, n)
        v1 = Vector(MPI.comm_world, n)
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
        v0 = Vector(MPI.comm_world, n)
        v1 = Vector(MPI.comm_world, n)
        v0[:] = -1.0
        v1[:] =  2.0
        v0 -= v1
        assert v0.sum() == -3.0*n

    def test_vector_assignment(self, any_backend):
        m, n = 301, 345
        v0 = Vector(MPI.comm_world, m)
        v1 = Vector(MPI.comm_world, n)
        v0[:] = -1.0
        v1[:] =  2.0
        v0 = v1
        assert v0.sum() == 2.0*n

    def test_vector_assignment_length(self, any_backend):
        # Test that assigning vectors of different lengths fails
        m, n = 301, 345
        v0 = Vector(MPI.comm_world, m)
        v1 = Vector(MPI.comm_world, n)
        def wrong_assignment(v0, v1):
            v0[:] = v1
        with pytest.raises(RuntimeError):
            wrong_assignment(v0, v1)

    def test_vector_assignment_length(self, any_backend):
        # Test that assigning with diffrent parallel layouts fails
        if MPI.size(MPI.comm_world) > 1:
            m = 301
            local_range0 = MPI.local_range(MPI.comm_world, m)
            print("local range", local_range0[0], local_range0[1])

            # Shift parallel partitiong but preserve global size
            if MPI.rank(MPI.comm_world) == 0:
                local_range1 = (local_range0[0], local_range0[1] + 1)
            elif MPI.rank(MPI.comm_world) == MPI.size(MPI.comm_world) - 1:
                local_range1 = (local_range0[0] + 1, local_range0[1])
            else:
                local_range1 = (local_range0[0] + 1, local_range0[1] + 1)

            v0 = Vector(MPI.comm_world)
            v0.init(local_range0)
            v1 = Vector(MPI.comm_world)
            v1.init(local_range1)
            assert v0.size() == v1.size()

            def wrong_assignment(v0, v1):
                v0[:] = v1
                with pytest.raises(RuntimeError):
                    wrong_assignment(v0, v1)


    # Test the access of the raw data through pointers
    # This is only available for Eigen backend
    def test_vector_data(self, data_backend):
        # Test for ordinary Vector
        v = Vector(MPI.comm_world, 301)
        v = as_backend_type(v)

        rw_array = v.array_view()
        assert rw_array.flags.owndata == False
        with pytest.raises(Exception):
            rw_array.resize([10])

        # Check that the array is a writable view
        rw_array[0] = 42
        ro_array = v.get_local()
        assert ro_array[0] == 42

        # Test for as_backend_type Vector
        v = as_backend_type(v)
        rw_array2 = v.array_view()
        assert (rw_array2 == ro_array).all()

    # xfail on TypeError
    xfail_type = pytest.mark.xfail(strict=True, raises=TypeError)
    xfail_type_py3 = pytest.mark.xfail(strict=True, raises=TypeError)


    @pytest.mark.parametrize("operand",
                             [int(42), 42.0, numpy.sin(1.0), numpy.float(42.0),
                                numpy.float64(42.0), numpy.float_(42.0),
                                numpy.int(42.0), numpy.long(42.0),
                                numpy.float16(42.0), numpy.float16(42.0),
                                numpy.float32(42.0), numpy.float128(42.0),
                                numpy.longfloat(42.0), numpy.int8(42.0),
                                numpy.int16(42.0), numpy.int32(42.0),
                                numpy.intc(42.0), numpy.longdouble(42.0),
                                numpy.int0(42.0), numpy.int64(42.0),
                                numpy.int_(42.0), numpy.longlong(42.0),
                             ])
    def test_vector_type_priority_with_numpy(self, any_backend, operand):
        """Test that DOLFIN return types are prefered over NumPy types for
        binary operations on NumPy objects

        """

        def _test_binary_ops(v, operand):
            assert isinstance(v + operand, cpp.la.GenericVector)
            assert isinstance(v - operand, cpp.la.GenericVector)
            assert isinstance(v*operand, cpp.la.GenericVector)
            assert isinstance(v/operand, cpp.la.GenericVector)
            assert isinstance(operand + v, cpp.la.GenericVector)
            assert isinstance(operand - v, cpp.la.GenericVector)
            assert isinstance(operand*v, cpp.la.GenericVector)
            assert isinstance(v+v, cpp.la.GenericVector)
            assert isinstance(v-v, cpp.la.GenericVector)
            assert isinstance(v*v, cpp.la.GenericVector)
            v += v.copy(); assert isinstance(v, cpp.la.GenericVector)
            v -= v.copy(); assert isinstance(v, cpp.la.GenericVector)
            v *= v.copy(); assert isinstance(v, cpp.la.GenericVector)
            v += operand; assert isinstance(v, cpp.la.GenericVector)
            v -= operand; assert isinstance(v, cpp.la.GenericVector)
            v *= operand; assert isinstance(v, cpp.la.GenericVector)
            v /= operand; assert isinstance(v, cpp.la.GenericVector)
            op = copy(operand); op += v; assert isinstance(op, cpp.la.GenericVector)
            op = copy(operand); op -= v; assert isinstance(op, cpp.la.GenericVector)
            op = copy(operand); op *= v; assert isinstance(op, cpp.la.GenericVector)

        # Test with vector wrapper
        v = Vector(MPI.comm_world, 8)
        _test_binary_ops(v, operand)

        # Test with vector casted to backend type
        v = as_backend_type(v)
        _test_binary_ops(v, operand)
