"""Unit tests for MPI facilities"""

# Copyright (C) 2017 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import sys

import mpi4py
import pytest
from dolfinx_utils.test.fixtures import tempdir  # noqa: F401
from mpi4py import MPI

import dolfinx
from dolfinx import wrappers
from dolfinx.jit import dolfinx_pc, mpi_jit_decorator


def test_mpi_comm_wrapper():
    """Test MPICommWrapper <-> mpi4py.MPI.Comm conversion"""
    w1 = MPI.COMM_WORLD
    m = dolfinx.UnitSquareMesh(w1, 4, 4)
    w2 = m.mpi_comm()
    assert isinstance(w1, MPI.Comm)
    assert isinstance(w2, MPI.Comm)


def test_mpi_comm_wrapper_cppimport(tempdir):  # noqa: F811
    """Test MPICommWrapper <-> mpi4py.MPI.Comm conversion for code compiled with cppimport"""

    cppimport = pytest.importorskip("cppimport")

    @mpi_jit_decorator
    def compile_module():
        cpp_code_header = f"""
        <%
        setup_pybind11(cfg)
        cfg['include_dirs'] += {dolfinx_pc["include_dirs"]
                                + [mpi4py.get_include()]
                                + [str(wrappers.get_include_path())]}
        %>
        """

        cpp_code = """
        #include <pybind11/pybind11.h>
        #include <caster_mpi.h>

        dolfinx_wrappers::MPICommWrapper
        test_comm_passing(const dolfinx_wrappers::MPICommWrapper comm)
        {
          MPI_Comm c = comm.get();
          return dolfinx_wrappers::MPICommWrapper(c);
        }

        PYBIND11_MODULE(test_mpi_comm_wrapper, m)
        {
            m.def("test_comm_passing", &test_comm_passing);
        }
        """

        open(os.path.join(tempdir, "test_mpi_comm_wrapper.cpp"), "w").write(cpp_code_header + cpp_code)

        sys.path.append(tempdir)
        return cppimport.imp("test_mpi_comm_wrapper")

    module = compile_module()

    w1 = MPI.COMM_WORLD
    w2 = module.test_comm_passing(w1)

    assert isinstance(w1, MPI.Comm)
    assert isinstance(w2, MPI.Comm)
    assert w1 == w2
