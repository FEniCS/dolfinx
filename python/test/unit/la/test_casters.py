"""Unit tests for PETSc casters"""

# Copyright (C) 2019 Francesco Ballarin
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pathlib

import numpy
import petsc4py
from dolfinx.jit import dolfinx_pc, mpi_jit_decorator
from dolfinx.wrappers import get_include_path as pybind_inc
from dolfinx_utils.test.fixtures import tempdir  # noqa: F401
from mpi4py import MPI
from petsc4py import PETSc
import cppimport


def test_petsc_casters_cppimport(tempdir):  # noqa: F811
    """Test casters of PETSc objects in codes compiled with cppimport"""

    @mpi_jit_decorator
    def compile_module():
        cpp_code_header = f"""
/*
<%
setup_pybind11(cfg)
cfg['include_dirs'] += {dolfinx_pc["include_dirs"] + [petsc4py.get_include()] + [str(pybind_inc())]}
cfg['compiler_args'] += {["-D" + dm for dm in dolfinx_pc["define_macros"]]}
cfg['libraries'] += {dolfinx_pc["libraries"]}
cfg['library_dirs'] += {dolfinx_pc["library_dirs"]}
%>
*/
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
PYBIND11_MODULE(petsc_casters_cppimport, m)
{
    m.def("PETSc_exp", &PETSc_exp);
}
"""

        path = pathlib.Path(tempdir)
        open(pathlib.Path(tempdir, "petsc_casters_cppimport.cpp"), "w").write(cpp_code + cpp_code_header)
        rel_path = path.relative_to(pathlib.Path(__file__).parent)
        p = str(rel_path).replace("/", ".") + ".petsc_casters_cppimport"
        return cppimport.imp(p)

    module = compile_module()

    # Define ranges
    comm = MPI.COMM_WORLD
    N = 10
    n = N // comm.size
    r = N % comm.size
    if comm.rank < r:
        local_range = [comm.rank * (n + 1), comm.rank * (n + 1) + n + 1]
    else:
        local_range = [comm.rank * n + r, comm.rank * n + r + n]

    # Create a PETSc vector
    x1 = PETSc.Vec()
    x1.create(MPI.COMM_WORLD)
    x1.setSizes((local_range[1] - local_range[0], None))
    x1.setFromOptions()
    x1.setArray(numpy.arange(local_range[0], local_range[1]))
    x2 = x1.copy()

    # Replace each component by its exponential
    module.PETSc_exp(x1)
    x2.exp()
    assert numpy.allclose(x1.getArray(), x2.getArray())
