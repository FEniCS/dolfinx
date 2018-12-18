// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_matrix_impl.h"
#include "DirichletBC.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "utils.h"
#include <Eigen/Sparse>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <petscsys.h>
#include <string>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
void fem::assemble_matrix(
    la::PETScMatrix& A, const Form& a,
    const std::vector<Eigen::SparseMatrix<PetscInt, Eigen::RowMajor>>&
        bc_dofs0,
    const std::vector<Eigen::SparseMatrix<PetscInt, Eigen::RowMajor>>&
        bc_dofs1,
    const std::vector<std::int32_t>& bc_dofs0_old,
    const std::vector<std::int32_t>& bc_dofs1_old)

{
  assert(!A.empty());

  assert(a.mesh());
  const mesh::Mesh& mesh = *a.mesh();

  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim);

  // Function spaces and dofmaps for each axis
  assert(a.function_space(0));
  assert(a.function_space(1));
  const function::FunctionSpace& V0 = *a.function_space(0);
  const function::FunctionSpace& V1 = *a.function_space(1);
  assert(V0.dofmap());
  assert(V1.dofmap());
  const fem::GenericDofMap& map0 = *V0.dofmap();
  const fem::GenericDofMap& map1 = *V1.dofmap();

  // Data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;

  // Iterate over all cells
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Check that cell is not a ghost
    assert(!cell.is_ghost());

    // Get cell vertex coordinates
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get dof maps for cell
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap0
        = map0.cell_dofs(cell.index());
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap1
        = map1.cell_dofs(cell.index());

    Ae.resize(dmap0.size(), dmap1.size());
    Ae.setZero();
    a.tabulate_tensor(Ae.data(), cell, coordinate_dofs);

    // FIXME: Pass in list  of cells, and list of local dofs, with
    // Dirichlet conditions
    // Note: could use negative dof indices to have PETSc do this
    // Zero rows/columns for Dirichlet bcs

    std::cout << "0:-----------------------------------" << std::endl;
    for (int i = 0; i < Ae.rows(); ++i)
    {
      const std::size_t ii = dmap0[i];
      if (std::find(bc_dofs0_old.begin(), bc_dofs0_old.end(), ii)
          != bc_dofs0_old.end())
      {
        std::cout << "Old cell, row, global: " << cell.index() << ", " << i
                  << ", " << ii << std::endl;
        Ae.row(i).setZero();
      }
    }
    // Loop over columns
    for (int j = 0; j < Ae.cols(); ++j)
    {
      const std::size_t jj = dmap1[j];
      if (std::find(bc_dofs1_old.begin(), bc_dofs1_old.end(), jj)
          != bc_dofs1_old.end())
        Ae.col(j).setZero();
    }

    for (auto& bc : bc_dofs0)
    {
    //   std::cout << "  vec of bcs" << std::endl;
      for (Eigen::SparseMatrix<PetscInt, Eigen::RowMajor>::InnerIterator it(
               bc, cell.index());
           it; ++it)
      {
        std::cout << "New cell, row, val: " << cell.index() << ", " << it.col()
                  << ", " << it.value() << std::endl;
        // Ae.row(it.col()).setZero();
      }
    }

    // for (auto& bc : bc_dofs1)
    // {
    //   for (Eigen::SparseMatrix<PetscScalar, Eigen::RowMajor>::InnerIterator
    //   it(
    //            bc, cell.index());
    //        it; ++it)
    //   {
    //     Ae.col(it.col()).setZero();
    //   }
    // }
    std::cout << "1:-----------------------------------" << std::endl;

    A.add_local(Ae.data(), dmap0.size(), dmap0.data(), dmap1.size(),
                dmap1.data());
  }
}
//-----------------------------------------------------------------------------
