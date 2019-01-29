# Copyright (C) 2011 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the JIT compiler"""

import numpy as np
import pytest

import dolfin
from dolfin import MPI, compile_cpp_code, cpp
from dolfin_utils.test.skips import skip_in_serial


def test_mpi_pybind11():
    """
    Test MPICommWrapper <-> mpi4py.MPI.Comm conversion for JIT-ed code
    """
    cpp_code = """
    #include <pybind11/pybind11.h>
    #include <dolfin_wrappers/MPICommWrapper.h>
    namespace dolfin
    {
      dolfin_wrappers::MPICommWrapper
      test_comm_passing(const dolfin_wrappers::MPICommWrapper comm)
      {
        MPI_Comm c = comm.get();
        return dolfin_wrappers::MPICommWrapper(c);
      }
    }
    PYBIND11_MODULE(SIGNATURE, m)
    {
        m.def("test_comm_passing", &dolfin::test_comm_passing);
    }
    """

    # Import MPI_COMM_WORLD
    from mpi4py import MPI
    w1 = MPI.COMM_WORLD

    # Compile the JIT module
    return pytest.xfail('Include path for dolfin_wrappers/* not set up to '
                        'work in the JIT at the moment')
    mod = dolfin.compile_cpp_code(cpp_code)

    # Pass a comm into C++ and get a new wrapper of the same comm back
    w2 = mod.test_comm_passing(w1)

    assert isinstance(w2, MPI.Comm)


def test_petsc():
    create_matrix_code = r'''
    #include <pybind11/pybind11.h>
    #include <dolfin.h>
    namespace dolfin
    {
        la::PETScMatrix create_matrix(void)
        {
            Mat I;
            la::PETScMatrix A = la::PETScMatrix(I);
            return A;
        }
    }

    PYBIND11_MODULE(SIGNATURE, m)
    {
      m.def("create_matrix", &dolfin::create_matrix);
    }
    '''
    module = compile_cpp_code(create_matrix_code)
    assert (module)


def test_pass_array_int():
    code = """
    #include <Eigen/Core>
    #include <pybind11/pybind11.h>
    #include <pybind11/eigen.h>
    using IntVecIn = Eigen::Ref<const Eigen::VectorXi>;
    int test_int_array(const IntVecIn arr)
    {
    return arr.sum();
    }
    PYBIND11_MODULE(SIGNATURE, m)
    {
    m.def("test_int_array", &test_int_array);
    }
    """
    module = compile_cpp_code(code)
    arr = np.array([1, 2, 4, 8], dtype=np.intc)
    ans = module.test_int_array(arr)
    assert ans == arr.sum() == 15


def test_pass_array_double():
    code = """
    #include <Eigen/Core>
    #include <pybind11/pybind11.h>
    #include <pybind11/eigen.h>
    using DoubleVecIn = Eigen::Ref<const Eigen::VectorXd>;
    int test_double_array(const DoubleVecIn arr)
    {
    return arr.sum();
    }
    PYBIND11_MODULE(SIGNATURE, m)
    {
    m.def("test_double_array", &test_double_array);
    }
    """
    module = compile_cpp_code(code)
    arr = np.array([1, 2, 4, 8], dtype=float)
    ans = module.test_double_array(arr)
    assert abs(arr.sum() - 15) < 1e-15
    assert abs(ans - 15) < 1e-15


@pytest.mark.skip(reason="FIXME: does this need pybind11 petsc_caster.h to be included?")
def test_compile_extension_module():
    # This test should do basically the same as the docstring of the
    # compile_extension_module function in compilemodule.py.  Remember
    # to update the docstring if the test is modified!

    code = """
      #include <pybind11/pybind11.h>
      #include <petscvec.h>
      #include <dolfin/la/PETScVector.h>

      void PETSc_exp(Vec x)
      {
        assert(x);
        VecExp(x);
      }

    PYBIND11_MODULE(SIGNATURE, m)
    {
      m.def("PETSc_exp", &PETSc_exp);
    }
    """

    ext_module = compile_cpp_code(code)
    local_range = MPI.local_range(MPI.comm_world, 10)
    x = cpp.la.create_vector(MPI.comm_world, local_range, [], 1)
    x_np = np.arange(float(local_range[1] - local_range[0]))
    with x.localForm() as lf:
        lf[:] = x_np

    ext_module.PETSc_exp(x)
    x_np = np.exp(x_np)

    x = x.getArray()
    assert (x == x_np).all()


@pytest.mark.xfail
def test_compile_extension_module_kwargs():
    # This test check that instant_kwargs of compile_extension_module
    # are taken into account when computing signature
    m2 = compile_cpp_code('', cppargs='-O2')
    m0 = compile_cpp_code('', cppargs='')
    assert not m2.__file__ == m0.__file__


@pytest.mark.skip
@skip_in_serial
def test_mpi_dependent_jiting():
    # FIXME: Not a proper unit test...
    from dolfin import (Expression, UnitSquareMesh, Function, TestFunction,
                        Form, FunctionSpace, dx, CompiledSubDomain,
                        SubSystemsManager)

    # Init petsc (needed to initalize petsc collectively on
    # all processes)
    SubSystemsManager.init_petsc()

    import mpi4py.MPI as mpi
    import petsc4py.PETSc as petsc

    # Set communicator and get process information
    comm = mpi.COMM_WORLD
    group = comm.Get_group()
    size = comm.Get_size()

    # Only consider parallel runs
    if size == 1:
        return

    rank = comm.Get_rank()
    group_comm_0 = petsc.Comm(comm.Create(group.Incl(range(1))))
    group_comm_1 = petsc.Comm(comm.Create(group.Incl(range(1, 2))))

    if size > 2:
        group_comm_2 = petsc.Comm(comm.Create(group.Incl(range(2, size))))

    if rank == 0:
        e = Expression("4", mpi_comm=group_comm_0, degree=0)

    elif rank == 1:
        e = Expression("5", mpi_comm=group_comm_1, degree=0)
        assert (e)
        domain = CompiledSubDomain(
            "on_boundary", mpi_comm=group_comm_1, degree=0)
        assert (domain)

    else:
        mesh = UnitSquareMesh(group_comm_2, 2, 2)
        V = FunctionSpace(mesh, "P", 1)
        u = Function(V)
        v = TestFunction(V)
        Form(u * v * dx)
