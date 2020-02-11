"""Unit tests for PETSc casters"""

# Copyright (C) 2019 Francesco Ballarin
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import sys

import numpy
import petsc4py
import pytest
from dolfinx_utils.test.fixtures import tempdir  # noqa: F401
from petsc4py import PETSc

import dolfinx
from dolfinx.wrappers import get_include_path as pybind_inc
from dolfinx.jit import dolfinx_pc, mpi_jit_decorator


def test_petsc_casters_cppimport(tempdir):  # noqa: F811
    """
    Test casters of PETSc objects in codes compiled with cppimport
    """

    cppimport = pytest.importorskip("cppimport")

    @mpi_jit_decorator
    def compile_module():
        cpp_code_header = f"""
        <%
        setup_pybind11(cfg)
        cfg['include_dirs'] += {dolfinx_pc["include_dirs"] + [petsc4py.get_include()] + [str(pybind_inc())]}
        cfg['compiler_args'] += {["-D" + dm for dm in dolfinx_pc["define_macros"]]}
        cfg['libraries'] += {dolfinx_pc["libraries"]}
        cfg['library_dirs'] += {dolfinx_pc["library_dirs"]}
        %>
        """

        cpp_code = """
        #include <pybind11/pybind11.h>
        #include <petscvec.h>
        #include <caster_petsc.h>

        void PETSc_exp(Vec x)
        {
          assert(x);
          VecExp(x);
        }
        PYBIND11_MODULE(test_petsc_casters_cppimport, m)
        {
            m.def("PETSc_exp", &PETSc_exp);
        }
        """

        open(os.path.join(tempdir, "test_petsc_casters_cppimport.cpp"), "w").write(cpp_code_header + cpp_code)

        sys.path.append(tempdir)
        return cppimport.imp("test_petsc_casters_cppimport")

    module = compile_module()

    # Create a PETSc vector
    local_range = dolfinx.MPI.local_range(dolfinx.MPI.comm_world, 10)
    x1 = PETSc.Vec()
    x1.create(dolfinx.MPI.comm_world)
    x1.setSizes((local_range[1] - local_range[0], None))
    x1.setFromOptions()
    x1.setArray(numpy.arange(local_range[0], local_range[1]))
    x2 = x1.copy()

    # Replace each component by its exponential
    module.PETSc_exp(x1)
    x2.exp()
    assert numpy.allclose(x1.getArray(), x2.getArray())
