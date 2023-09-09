# Copyright (C) 2018-2019 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly using cppimport"""

import pathlib

import cppimport
import dolfinx.pkgconfig
import numpy as np
import pybind11
import pytest
import scipy.sparse.linalg
import ufl
from dolfinx.fem import (FunctionSpace, assemble_matrix, dirichletbc, form,
                         locate_dofs_geometrical)
from dolfinx.mesh import create_unit_square
from dolfinx.wrappers import get_include_path as pybind_inc
from mpi4py import MPI

import dolfinx


@pytest.mark.skip_in_parallel
@pytest.mark.skipif(not dolfinx.pkgconfig.exists("eigen3"),
                    reason="This test needs eigen3 pkg-config.")
@pytest.mark.skipif(not dolfinx.pkgconfig.exists("dolfinx"),
                    reason="This test needs DOLFINx pkg-config.")
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_eigen_assembly(tempdir, dtype):  # noqa: F811
    """Compare assembly into scipy.CSR matrix with native assembly"""
    def compile_eigen_csr_assembler_module():
        dolfinx_pc = dolfinx.pkgconfig.parse("dolfinx")
        eigen_dir = dolfinx.pkgconfig.parse("eigen3")["include_dirs"]
        cpp_code_header = f"""
<%
setup_pybind11(cfg)
cfg['include_dirs'] = {dolfinx_pc["include_dirs"]
  + [pybind11.get_include()] + [str(pybind_inc())] + eigen_dir}
cfg['compiler_args'] = ["-std=c++20", "-Wno-comment"]
cfg['libraries'] = {dolfinx_pc["libraries"]}
cfg['library_dirs'] = {dolfinx_pc["library_dirs"]}
%>
"""

        cpp_typemap = {np.float64: "double",
                       np.float32: "float",
                       np.complex64: "std::complex<float>",
                       np.complex128: "std::complex<double>"}

        cpp_code = f"""
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <vector>
#include <complex>
#include <Eigen/Sparse>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>

template<typename T>
Eigen::SparseMatrix<T, Eigen::RowMajor>
assemble_csr(const dolfinx::fem::Form<T>
& a,
             const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs)
{{
  std::vector<Eigen::Triplet<T>> triplets;
  auto mat_add
      = [&triplets](const std::span<const std::int32_t>& rows,
                    const std::span<const std::int32_t>& cols,
                    const std::span<const T>& v)
    {{
      for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t j = 0; j < cols.size(); ++j)
          triplets.emplace_back(rows[i], cols[j], v[i * cols.size() + j]);
      return 0;
    }};

  dolfinx::fem::assemble_matrix(mat_add, a, bcs);

  auto map0 = a.function_spaces().at(0)->dofmap()->index_map;
  int bs0 = a.function_spaces().at(0)->dofmap()->index_map_bs();
  auto map1 = a.function_spaces().at(1)->dofmap()->index_map;
  int bs1 = a.function_spaces().at(1)->dofmap()->index_map_bs();
  Eigen::SparseMatrix<T, Eigen::RowMajor> mat(
      bs0 * (map0->size_local() + map0->num_ghosts()),
      bs1 * (map1->size_local() + map1->num_ghosts()));
  mat.setFromTriplets(triplets.begin(), triplets.end());
  return mat;
}}

PYBIND11_MODULE(eigen_csr, m)
{{
  m.def("assemble_matrix", &assemble_csr<{cpp_typemap[dtype]}>);
}}
"""

        path = pathlib.Path(tempdir)
        open(pathlib.Path(tempdir, "eigen_csr.cpp"), "w").write(cpp_code + cpp_code_header)
        rel_path = path.relative_to(pathlib.Path(__file__).parent)
        p = str(rel_path).replace("/", ".") + ".eigen_csr"
        return cppimport.imp(p)

    def assemble_csr_matrix(a, bcs):
        """Assemble bilinear form into an SciPy CSR matrix, in serial."""
        module = compile_eigen_csr_assembler_module()
        A = module.assemble_matrix(a, bcs)
        if a.function_spaces[0] is a.function_spaces[1]:
            for bc in bcs:
                if a.function_spaces[0].contains(bc.function_space):
                    bc_dofs, _ = bc.dof_indices()
                    # See https://github.com/numpy/numpy/issues/14132
                    # for why we copy bc_dofs as a work-around
                    dofs = bc_dofs.copy()
                    A[dofs, dofs] = 1.0
        return A

    realtype = np.real(dtype(1.0)).dtype
    mesh = create_unit_square(MPI.COMM_SELF, 12, 12, dtype=realtype)
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(Q)
    v = ufl.TestFunction(Q)

    a = form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, dtype=dtype)

    bdofsQ = locate_dofs_geometrical(Q, lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)))
    bc = dirichletbc(dtype(1), bdofsQ, Q)

    A1 = assemble_matrix(a, [bc])
    A1.scatter_reverse()
    A2 = assemble_csr_matrix(a._cpp_object, [bc._cpp_object])
    assert np.isclose(np.sqrt(A1.squared_norm()), scipy.sparse.linalg.norm(A2))


@pytest.mark.skip_in_parallel
@pytest.mark.skipif(not dolfinx.pkgconfig.exists("dolfinx"),
                    reason="This test needs DOLFINx pkg-config.")
def test_csr_assembly(tempdir):  # noqa: F811

    def compile_assemble_csr_assembler_module(a_form):
        dolfinx_pc = dolfinx.pkgconfig.parse("dolfinx")
        cpp_code_header = f"""
<%
setup_pybind11(cfg)
cfg['include_dirs'] = {dolfinx_pc["include_dirs"]
  + [pybind11.get_include()] + [str(pybind_inc())] }
cfg['compiler_args'] = ["-std=c++20", "-Wno-comment"]
cfg['libraries'] = {dolfinx_pc["libraries"]}
cfg['library_dirs'] = {dolfinx_pc["library_dirs"]}
%>
"""
        if dolfinx.default_scalar_type == np.float32:
            dtype = "float"
        elif dolfinx.default_scalar_type == np.float64:
            dtype = "double"
        elif dolfinx.default_scalar_type == np.complex64:
            dtype = "std::complex<float>"
        elif dolfinx.default_scalar_type == np.complex128:
            dtype = "std::complex<double>"
        else:
            raise RuntimeError("Unknown scalar type")

        bs = [a_form.function_spaces[0].dofmap.index_map_bs,
              a_form.function_spaces[1].dofmap.index_map_bs]

        cpp_code = f"""
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/la/MatrixCSR.h>

template<typename T>
dolfinx::la::MatrixCSR<T>
assemble_csr(const dolfinx::fem::Form<T>& a,
             const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs)
{{
  dolfinx::la::SparsityPattern sp = create_sparsity_pattern(a);
  sp.finalize();
  dolfinx::la::MatrixCSR<T> A(sp);
  auto mat_add = A.template mat_add_values<{bs[0]}, {bs[1]}>();
  dolfinx::fem::assemble_matrix(mat_add, a, bcs);
  return A;
}}

PYBIND11_MODULE(assemble_csr, m)
{{
  m.def("assemble_matrix_bs", &assemble_csr<{dtype}>);
}}
"""
        path = pathlib.Path(tempdir)
        open(pathlib.Path(tempdir, "assemble_csr.cpp"), "w").write(cpp_code + cpp_code_header)
        rel_path = path.relative_to(pathlib.Path(__file__).parent)
        p = str(rel_path).replace("/", ".") + ".assemble_csr"
        return cppimport.imp(p)

    mesh = create_unit_square(MPI.COMM_SELF, 11, 7)
    gdim = mesh.geometry.dim
    Q = FunctionSpace(mesh, ("Lagrange", 1, (gdim,)))
    Q2 = FunctionSpace(mesh, ("Lagrange", 1), (3,))
    u = ufl.TrialFunction(Q)
    v = ufl.TestFunction(Q2)

    a = form(ufl.inner(ufl.grad(u[0]), ufl.grad(v[0])) * ufl.dx)

    module = compile_assemble_csr_assembler_module(a)
    A = module.assemble_matrix_bs(a._cpp_object, [])
    assert np.isclose(A.squared_norm(), 1743.3479507505413)
