"""Unit tests for MPI facilities"""

# Copyright (C) 2017 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pathlib

import cppimport
import pytest

import dolfinx
import dolfinx.pkgconfig
from dolfinx import wrappers
from dolfinx.jit import mpi_jit_decorator
from dolfinx.mesh import create_unit_square
import nanobind

import mpi4py
from mpi4py import MPI


def test_mpi_comm_wrapper():
    """Test MPICommWrapper <-> mpi4py.MPI.Comm conversion"""
    w1 = MPI.COMM_WORLD
    m = create_unit_square(w1, 4, 4)
    w2 = m.comm
    assert isinstance(w1, MPI.Comm)
    assert isinstance(w2, MPI.Comm)


# @pytest.mark.skipif(not dolfinx.pkgconfig.exists("dolfinx"),
#                     reason="This test needs DOLFINx pkg-config.")
# def test_mpi_comm_wrapper_cppimport(tempdir):  # noqa: F811
#     """Test MPICommWrapper <-> mpi4py.MPI.Comm conversion for code compiled with cppimport"""

#     dolfinx_pc = dolfinx.pkgconfig.parse("dolfinx")

#     @mpi_jit_decorator
#     def compile_module():
#         cpp_code_header = f"""
# /*
# <%
# import nanobind
# cfg['compiler_args'] = ['-std=c++20']
# cfg['include_dirs'] += {dolfinx_pc["include_dirs"]
#                         + [mpi4py.get_include()]
#                         + [nanobind.include_dir()]
#                         + [str(wrappers.get_include_path())]}
# %>
# */
# """

#         cpp_code = """
# #include <nanobind/nanobind.h>
# #include <caster_mpi.h>

# dolfinx_wrappers::MPICommWrapper
# test_comm_passing(const dolfinx_wrappers::MPICommWrapper comm)
# {
#     MPI_Comm c = comm.get();
#     return dolfinx_wrappers::MPICommWrapper(c);
# }

# NB_MODULE(mpi_comm_wrapper, m)
# {
#     m.def("test_comm_passing", &test_comm_passing);
# }
# """

#         path = pathlib.Path(tempdir)
#         open(pathlib.Path(tempdir, "mpi_comm_wrapper.cpp"), "w").write(cpp_code + cpp_code_header)
#         rel_path = path.relative_to(pathlib.Path(__file__).parent)
#         p = str(rel_path).replace("/", ".") + ".mpi_comm_wrapper"
#         return cppimport.imp(p)

#     module = compile_module(MPI.COMM_WORLD)

#     w1 = MPI.COMM_WORLD
#     w2 = module.test_comm_passing(w1)

#     assert isinstance(w1, MPI.Comm)
#     assert isinstance(w2, MPI.Comm)
#     assert w1 == w2
