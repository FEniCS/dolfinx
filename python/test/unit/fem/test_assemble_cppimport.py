# Copyright (C) 2018-2019 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly using cppimport"""

import pathlib

import cppimport
import dolfinx
import numpy
import petsc4py
import scipy.sparse.linalg
import ufl
from dolfinx.generation import UnitSquareMesh
from dolfinx.jit import dolfinx_pc
from dolfinx.wrappers import get_include_path as pybind_inc
from dolfinx_utils.test.fixtures import tempdir  # noqa: F401
from dolfinx_utils.test.skips import skip_in_parallel
from mpi4py import MPI


@skip_in_parallel
def test_eigen_assembly(tempdir):  # noqa: F811
    """Compare assembly into scipy.CSR matrix with PETSc assembly"""
    def compile_eigen_csr_assembler_module():
        cpp_code_header = f"""
<%
setup_pybind11(cfg)
cfg['include_dirs'] = {dolfinx_pc["include_dirs"] + [petsc4py.get_include()] + [str(pybind_inc())]}
cfg['compiler_args'] = {["-D" + dm for dm in dolfinx_pc["define_macros"]]}
cfg['compiler_args'] = ['-std=c++17']
cfg['libraries'] = {dolfinx_pc["libraries"]}
cfg['library_dirs'] = {dolfinx_pc["library_dirs"]}
%>
"""

        cpp_code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <vector>
#include <Eigen/Sparse>
#include <petscsys.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>

template<typename T>
Eigen::SparseMatrix<T, Eigen::RowMajor>
assemble_csr(const dolfinx::fem::Form<T>& a,
        const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs)
{
std::vector<Eigen::Triplet<T>> triplets;
const auto mat_add
    = [&triplets](std::int32_t nrow, const std::int32_t* rows,
                std::int32_t ncol, const std::int32_t* cols, const T* v)
    {
    for (int i = 0; i < nrow; ++i)
        for (int j = 0; j < ncol; ++j)
        triplets.emplace_back(rows[i], cols[j], v[i * ncol + j]);
    return 0;
    };

dolfinx::fem::assemble_matrix<T>(mat_add, a, bcs);

auto map0 = a.function_space(0)->dofmap()->index_map;
auto map1 = a.function_space(1)->dofmap()->index_map;
Eigen::SparseMatrix<T, Eigen::RowMajor> mat(
    map0->block_size() * (map0->size_local() + map0->num_ghosts()),
    map1->block_size() * (map1->size_local() + map1->num_ghosts()));
mat.setFromTriplets(triplets.begin(), triplets.end());
return mat;
}

PYBIND11_MODULE(eigen_csr, m)
{
m.def("assemble_matrix", &assemble_csr<PetscScalar>);
}
"""

        path = pathlib.Path(tempdir)
        open(pathlib.Path(tempdir, "eigen_csr.cpp"), "w").write(cpp_code + cpp_code_header)
        rel_path = path.relative_to(pathlib.Path(__file__).parent)
        p = str(rel_path).replace("/", ".") + ".eigen_csr"
        return cppimport.imp(p)

    def assemble_csr_matrix(a, bcs):
        """Assemble bilinear form into an SciPy CSR matrix, in serial."""
        module = compile_eigen_csr_assembler_module()
        _a = dolfinx.fem.assemble._create_cpp_form(a)
        A = module.assemble_matrix(_a, bcs)
        if _a.function_spaces[0].id == _a.function_spaces[1].id:
            for bc in bcs:
                if _a.function_spaces[0].contains(bc.function_space):
                    bc_dofs = bc.dof_indices[:, 0]
                    A[bc_dofs, bc_dofs] = 1.0
        return A

    mesh = UnitSquareMesh(MPI.COMM_SELF, 12, 12)
    Q = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(Q)
    v = ufl.TestFunction(Q)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    bdofsQ = dolfinx.fem.locate_dofs_geometrical(Q, lambda x: numpy.logical_or(x[0] < 1.0e-6, x[0] > 1.0 - 1.0e-6))
    u_bc = dolfinx.function.Function(Q)
    with u_bc.vector.localForm() as u_local:
        u_local.set(1.0)
    bc = dolfinx.fem.dirichletbc.DirichletBC(u_bc, bdofsQ)

    A1 = dolfinx.fem.assemble_matrix(a, [bc])
    A1.assemble()
    A2 = assemble_csr_matrix(a, [bc])
    assert numpy.isclose(A1.norm(), scipy.sparse.linalg.norm(A2))
