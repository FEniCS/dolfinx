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
    const Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& dmap0x,
    const Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& dmap1x,
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

    const std::size_t cell_index = cell.index();

    // Get dof maps for cell
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap0
        = map0.cell_dofs(cell_index);
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap1
        = map1.cell_dofs(cell_index);

    // std::cout << "c0:-----------------------------------" << std::endl;
    // std::cout << dmap0 << std::endl;
    // std::cout << "--------------------------------------" << std::endl;
    // std::cout << dmap0x.row(cell_index) << std::endl;
    // std::cout << dmap0x.row(cell_index).size() << std::endl;
    // std::cout << *(dmap0x.row(cell_index).data() + 1) << std::endl;
    // std::cout << "c1:-----------------------------------" << std::endl;

    Ae.resize(dmap0.size(), dmap1.size());
    Ae.setZero();
    a.tabulate_tensor(Ae.data(), cell, coordinate_dofs);

    // FIXME: Pass in list  of cells, and list of local dofs, with
    // Dirichlet conditions
    // Note: could use negative dof indices to have PETSc do this
    // Zero rows/columns for Dirichlet bcs

    // std::cout << "0:-----------------------------------" << std::endl;
    for (int i = 0; i < Ae.rows(); ++i)
    {
      const std::int32_t ii = dmap0[i];
      // std::cout << "dmap check (0): " << dmap0x(cell.index(), i) << ", " << ii
      //           << std::endl;
      assert(ii == dmap0x(cell_index, i) or ii == -dmap0x(cell_index, i) - 1);
      if (std::find(bc_dofs0_old.begin(), bc_dofs0_old.end(), ii)
          != bc_dofs0_old.end())
      {
        // std::cout << "   Row old, global: " << cell.index() << ", " << i << ", "
        //           << ii << std::endl;
        assert(dmap0x(cell_index, i) < 0);
        // Ae.row(i).setZero();
      }
      else
      {
        assert(dmap0x(cell.index(), i) >= 0);
        // std::cout << "   No bc applied" << std::endl;
      }
    }
    // std::cout << "1:-----------------------------------" << std::endl;
    // std::cout << dmap1 << std::endl;
    // std::cout << dmap1x.row(cell_index) << std::endl;

    // Loop over columns
    for (int j = 0; j < Ae.cols(); ++j)
    {
      const std::int32_t jj = dmap1[j];
      assert(jj == dmap1x(cell_index, j) or jj == -dmap1x(cell_index, j) - 1);
      if (std::find(bc_dofs1_old.begin(), bc_dofs1_old.end(), jj)
          != bc_dofs1_old.end())
      {
        // std::cout << "   dof on: " << j << ", " << jj << std::endl;
        assert(dmap1x(cell_index, j) < 0);
        // Ae.col(j).setZero();
      }
      else
      {
        assert(dmap1x(cell.index(), j) >= 0);
      }
    }

    // for (auto& bc : bc_dofs0)
    // {
    //   //   std::cout << "  vec of bcs" << std::endl;
    //   for (Eigen::SparseMatrix<PetscInt, Eigen::RowMajor>::InnerIterator it(
    //            bc, cell.index());
    //        it; ++it)
    //   {
    //     std::cout << "New cell, row, val: " << cell.index() << ", " <<
    //     it.col()
    //               << ", " << it.value() << std::endl;
    //     // Ae.row(it.col()).setZero();
    //   }
    // }

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
    // std::cout << "2:-----------------------------------" << std::endl;

    // A.add_local(Ae.data(), dmap0x.row(cell_index).size(),
    //             dmap0x.row(cell_index).data(), dmap1x.row(cell_index).size(),
    //             dmap1x.row(cell_index).data());
    A.add_local(Ae.data(), dmap0.size(), dmap0.data(), dmap1.size(),
                dmap1.data());
  }
}
//-----------------------------------------------------------------------------
