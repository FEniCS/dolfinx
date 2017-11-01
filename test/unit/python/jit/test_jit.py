"""Unit tests for the JIT compiler"""

# Copyright (C) 2011 Anders Logg
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

import pytest
import platform
import dolfin
from dolfin import *
from dolfin_utils.test import (skip_if_not_PETSc, skip_if_not_SLEPc,
                               skip_if_not_MPI, skip_in_serial,
                               skip_if_not_petsc4py, skip_if_pybind11,
                               skip_if_not_pybind11)


def test_nasty_jit_caching_bug():

    # This may result in something like "matrices are not aligned"
    # from FIAT if the JIT caching does not recognize that the two
    # forms are different

    default_parameters = parameters["form_compiler"]["representation"]
    for representation in ["quadrature"]:

        parameters["form_compiler"]["representation"] = representation

        M1 = assemble(Constant(1.0)*dx(UnitSquareMesh(4, 4)))
        M2 = assemble(Constant(1.0)*dx(UnitCubeMesh(4, 4, 4)))

        assert round(M1 - 1.0, 7) == 0
        assert round(M2 - 1.0, 7) == 0

    parameters["form_compiler"]["representation"] = default_parameters


@skip_if_pybind11
@skip_if_not_MPI
def test_mpi_swig():
    create_transfer_matrix_code = r'''
    namespace dolfin
    {
        void find_exterior_points(MPI_Comm mpi_comm) {}
    }'''
    compile_extension_module(code=create_transfer_matrix_code)


@skip_if_not_pybind11
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
    if dolfin.has_mpi4py():
        from mpi4py import MPI
        w1 = MPI.COMM_WORLD
    else:
        w1 = dolfin.MPI.comm_world

    # Compile the JIT module
    return pytest.xfail('Include path for dolfin_wrappers/* not set up to '
                        'work in the JIT at the moment')
    mod = dolfin.compile_cpp_code(cpp_code)

    # Pass a comm into C++ and get a new wrapper of the same comm back
    w2 = mod.test_comm_passing(w1)

    if dolfin.has_mpi4py():
        assert isinstance(w2, MPI.Comm)
    else:
        assert isinstance(w2, dolfin.cpp.MPICommWrapper)
        assert w1.underlying_comm() == w2.underlying_comm()


@skip_if_pybind11
@skip_if_not_PETSc
def test_pesc_swig():
    create_matrix_code = r'''
    namespace dolfin
    {
        std::shared_ptr<PETScMatrix> create_matrix(void) {
            Mat I;
            std::shared_ptr<PETScMatrix> ptr = std::make_shared<PETScMatrix>(I);
            return ptr;
        }
    }
    '''
    compile_extension_module(code=create_matrix_code)


@skip_if_pybind11
@skip_if_not_SLEPc
def test_slepc_swig():
    create_eps_code = r'''
    #include <slepc.h>
    namespace dolfin
    {
        std::shared_ptr<EPS> create_matrix(MPI_Comm comm) {
            EPS eps;
            EPSCreate(comm, &eps);
            std::shared_ptr<EPS> ptr = std::make_shared<EPS>(eps);
            return ptr;
        }
    }
    '''
    compile_extension_module(code=create_eps_code)


def test_pass_array_int():
    import numpy

    if has_pybind11():
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
    else:
        code = """
        int test_int_array(const Array<int>& int_arr)
        {
            int ret = 0;
            for (int i = 0; i < int_arr.size(); i++)
            {
                ret += int_arr[i];
            }
            return ret;
        }
        """
        module = compile_extension_module(code=code,
                                          source_directory='.',
                                          sources=[],
                                          include_dirs=["."])
    arr = numpy.array([1, 2, 4, 8], dtype=numpy.intc)
    ans = module.test_int_array(arr)
    assert ans == arr.sum() == 15


def test_pass_array_double():
    import numpy

    if has_pybind11():
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
    else:
        code = """
        double test_double_array(const Array<double>& arr)
        {
            double ret = 0;
            for (int i = 0; i < arr.size(); i++)
            {
                ret += arr[i];
            }
            return ret;
        }
        """
        module = compile_extension_module(code=code,
                                          source_directory='.',
                                          sources=[],
                                          include_dirs=["."])
    arr = numpy.array([1, 2, 4, 8], dtype=float)
    ans = module.test_double_array(arr)
    assert abs(arr.sum() - 15) < 1e-15
    assert abs(ans - 15) < 1e-15


@skip_if_pybind11
@skip_if_not_PETSc
def test_compile_extension_module():

    # This test should do basically the same as the docstring of the
    # compile_extension_module function in compilemodule.py.  Remember
    # to update the docstring if the test is modified!

    from numpy import arange, exp
    code = """
    namespace dolfin {

      void PETSc_exp(std::shared_ptr<dolfin::PETScVector> vec)
      {
        Vec x = vec->vec();
        assert(x);
        VecExp(x);
      }
    }
    """
    for module_name in ["mypetscmodule_" + dolfin.__version__ +
                        "_py-" + platform.python_version()[:3], ""]:
        ext_module = compile_extension_module(\
            code, module_name=module_name,\
            additional_system_headers=["petscvec.h"])
        vec = PETScVector(mpi_comm_world(), 10)
        np_vec = vec.array()
        np_vec[:] = arange(len(np_vec))
        vec.set_local(np_vec)
        ext_module.PETSc_exp(vec)
        np_vec[:] = exp(np_vec)
        assert (np_vec == vec.array()).all()


@skip_if_not_pybind11
@skip_if_not_PETSc
def test_compile_extension_module_pybind11():

    # This test should do basically the same as the docstring of the
    # compile_extension_module function in compilemodule.py.  Remember
    # to update the docstring if the test is modified!

    from numpy import arange, exp
    code = """
      #include <pybind11/pybind11.h>

      #include <petscvec.h>
      #include <dolfin/la/PETScVector.h>

      void PETSc_exp(std::shared_ptr<dolfin::PETScVector> vec)
      {
        Vec x = vec->vec();
        assert(x);
        VecExp(x);
      }

    PYBIND11_MODULE(SIGNATURE, m)
    {
      m.def("PETSc_exp", &PETSc_exp);
    }
    """

    ext_module = compile_cpp_code(code)

    vec = PETScVector(mpi_comm_world(), 10)
    np_vec = vec.get_local()
    np_vec[:] = arange(len(np_vec))
    vec.set_local(np_vec)
    ext_module.PETSc_exp(vec)
    np_vec[:] = exp(np_vec)
    assert (np_vec == vec.get_local()).all()


@skip_if_pybind11
def test_compile_extension_module_kwargs():
    # This test check that instant_kwargs of compile_extension_module
    # are taken into account when computing signature
    m2 = compile_extension_module('', cppargs='-O2')
    m0 = compile_extension_module('', cppargs='')
    assert not m2.__file__ == m0.__file__


@skip_if_pybind11
@skip_if_not_petsc4py
@skip_in_serial
def test_mpi_dependent_jiting():
    # FIXME: Not a proper unit test...
    from dolfin import (Expression, UnitSquareMesh, Function,
                        TestFunction, Form, FunctionSpace, dx, CompiledSubDomain,
                        SubSystemsManager)

    # Init petsc (needed to initalize petsc and slepc collectively on
    # all processes)
    SubSystemsManager.init_petsc()

    try:
        import mpi4py.MPI as mpi
    except:
        return

    try:
        import petsc4py.PETSc as petsc
    except:
        return

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
        group_comm_2 = petsc.Comm(comm.Create(group.Incl(range(2,size))))

    if rank == 0:
        e = Expression("4", mpi_comm=group_comm_0, degree=0)

    elif rank == 1:
        e = Expression("5", mpi_comm=group_comm_1, degree=0)
        domain = CompiledSubDomain("on_boundary", mpi_comm=group_comm_1,
                                   degree=0)

    else:
        mesh = UnitSquareMesh(group_comm_2, 2, 2)
        V = FunctionSpace(mesh, "P", 1)
        u = Function(V)
        v = TestFunction(V)
        Form(u*v*dx)
