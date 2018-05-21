// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Assembler.h"
#include "DirichletBC.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "SparsityPatternBuilder.h"
#include "utils.h"
#include <dolfin/common/types.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <string>

#include <petscis.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
Assembler::Assembler(std::vector<std::vector<std::shared_ptr<const Form>>> a,
                     std::vector<std::shared_ptr<const Form>> L,
                     std::vector<std::shared_ptr<const DirichletBC>> bcs)
    : _a(a), _l(L), _bcs(bcs)
{
  assert(!a.empty());
  assert(!a[0].empty());

  // FIXME: check that a is square
  // FIXME: a.size() = L.size()
  // FIXME: check ranks
  // FIXME: check that function spaces in the blocks match, and are not
  //        repeated
  // FIXME: figure out number or blocks (row and column)
}
//-----------------------------------------------------------------------------
void Assembler::assemble(la::PETScMatrix& A, BlockType block_type)
{
  // Check if matrix should be nested
  assert(!_a.empty());
  const bool block_matrix = _a.size() > 1 or _a[0].size() > 1;

  if (A.empty())
  {
    std::vector<std::vector<const Form*>> forms(
        _a.size(), std::vector<const Form*>(_a[0].size()));
    for (std::size_t i = 0; i < _a.size(); ++i)
      for (std::size_t j = 0; j < _a[i].size(); ++j)
        forms[i][j] = _a[i][j].get();

    // Initialise matrix
    if (block_type == BlockType::nested)
    {
      std::cout << "Init MatNest" << std::endl;
      fem::init_nest(A, forms);
    }
    else if (block_matrix and block_type == BlockType::monolithic)
      fem::init_monolithic(A, forms);
    else
      init(A, *_a[0][0]);
  }

  // Get PETSc matrix type
  MatType mat_type;
  MatGetType(A.mat(), &mat_type);
  const bool is_matnest = strcmp(mat_type, MATNEST) == 0 ? true : false;

  // Assemble matrix
  if (is_matnest)
  {
    for (std::size_t i = 0; i < _a.size(); ++i)
    {
      for (std::size_t j = 0; j < _a[i].size(); ++j)
      {
        // Get submatrix
        Mat subA;
        MatNestGetSubMat(A.mat(), i, j, &subA);
        if (_a[i][j])
        {
          la::PETScMatrix mat(subA);
          this->assemble(mat, *_a[i][j], _bcs);
          // mat.apply(la::PETScMatrix::AssemblyType::FINAL);
        }
        else
        {
          // FIXME: Figure out how to check that matrix block is null
          // Null block, do nothing
        }
      }
    }
  }
  else if (block_matrix)
  {
    // MPI::barrier(MPI_COMM_WORLD);
    // std::cout << "PPPPPPPPPPPPPPPPPP: " << std::endl;
    // ISLocalToGlobalMapping rmap, cmap;
    // MatGetLocalToGlobalMapping(A.mat(), &rmap, &cmap);

    // if (MPI::rank(MPI_COMM_WORLD) == 0)
    //   ISLocalToGlobalMappingView(rmap, PETSC_VIEWER_STDOUT_SELF);

    // std::cout << "QQQQPPPPPPPPPPPPPPPPP: " << std::endl;
    // MPI::barrier(MPI_COMM_WORLD);
    // MPI_Comm mpi_comm = MPI_COMM_NULL;
    // PetscObjectGetComm((PetscObject)rmap, &mpi_comm);
    // std::cout << "Comm size: " << MPI::size(mpi_comm);
    // MPI::barrier(MPI_COMM_WORLD);

    // exit(0);

    std::vector<std::pair<la_index_t, double>> bc_values;

    // MPI::barrier(MPI_COMM_WORLD);
    // MPI::barrier(MPI_COMM_WORLD);
    // for (std::size_t i = 0; i < _a.size(); ++i)
    // {
    //   for (std::size_t j = 0; j < _a[i].size(); ++j)
    //   {
    //     auto map0 = _a[i][j]->function_space(0)->dofmap()->index_map();
    //     if (MPI::rank(MPI_COMM_WORLD) == 1)
    //     {
    //       std::cout << "Owned sizes rows: "
    //                 << map0->size(common::IndexMap::MapSize::OWNED)
    //                 << std::endl;
    //     }
    //   }
    // }
    // MPI::barrier(MPI_COMM_WORLD);
    // MPI::barrier(MPI_COMM_WORLD);

    // MPI::barrier(MPI_COMM_WORLD);
    std::int64_t offset_row = 0;
    std::cout << "**** mat size: " << A.size()[0] << ", " << A.size()[0]
              << std::endl;
    for (std::size_t i = 0; i < _a.size(); ++i)
    {
      // Loop over columns
      std::int64_t offset_col = 0;
      for (std::size_t j = 0; j < _a[i].size(); ++j)
      {
        if (_a[i][j])
        {
          // MPI::barrier(MPI_COMM_WORLD);
          // MPI::barrier(MPI_COMM_WORLD);
          // std::cout << "--- AAAAAAAAAAAAAAAAa: " << i << ", " << j <<
          // std::endl; MPI::barrier(MPI_COMM_WORLD);
          // MPI::barrier(MPI_COMM_WORLD);

          // Build index set for block
          auto map0 = _a[i][j]->function_space(0)->dofmap()->index_map();
          auto map1 = _a[i][j]->function_space(1)->dofmap()->index_map();
          // auto map0_size = map0->size(common::IndexMap::MapSize::OWNED);
          // auto map1_size = map1->size(common::IndexMap::MapSize::OWNED);
          auto map0_size = map0->size(common::IndexMap::MapSize::ALL);
          auto map1_size = map1->size(common::IndexMap::MapSize::ALL);

          std::vector<PetscInt> index0(map0_size);
          std::vector<PetscInt> index1(map1_size);
          std::iota(index0.begin(), index0.end(), offset_row);
          std::iota(index1.begin(), index1.end(), offset_col);

          IS is0, is1;
          ISCreateBlock(MPI_COMM_SELF, map0->block_size(), index0.size(),
                        index0.data(), PETSC_COPY_VALUES, &is0);
          ISCreateBlock(MPI_COMM_SELF, map1->block_size(), index1.size(),
                        index1.data(), PETSC_COPY_VALUES, &is1);
          // if (MPI::rank(MPI_COMM_WORLD) == 0)
          // {
          //   // std::cout << "Owned sizes rows: "
          //   //           << map0->size(common::IndexMap::MapSize::OWNED)
          //   //           << std::endl;
          //   std::cout << "Field: " << i << ", " << j << std::endl;
          //   std::cout << "  Rows:" << std::endl;
          //   for (auto x : index0)
          //     std::cout << "   " << x << std::endl;
          //   std::cout << "  Cols:" << std::endl;
          //   for (auto x : index1)
          //     std::cout << "   " << x << std::endl;
          // }

          // Get sub-matrix (using local indices for is0 and is1)
          Mat subA;
          // MPI::barrier(MPI_COMM_WORLD);
          // std::cout << "Get block: " << MPI::rank(MPI_COMM_WORLD) <<
          // std::endl;
          MatGetLocalSubMatrix(A.mat(), is0, is1, &subA);
          // std::cout << "Post get block: " << MPI::rank(MPI_COMM_WORLD)
          //           << std::endl;
          // MPI::barrier(MPI_COMM_WORLD);

          // ISLocalToGlobalMapping rmap, cmap;
          // MatGetLocalToGlobalMapping(subA, &rmap, &cmap);
          // if (MPI::rank(MPI_COMM_WORLD) == 1)
          // {
          //   std::cout << "**************: " << i << ", " << j << std::endl;
          //   ISLocalToGlobalMappingView(rmap, PETSC_VIEWER_STDOUT_SELF);
          //   std::cout << "+++++ " << std::endl;
          //   ISLocalToGlobalMappingView(cmap, PETSC_VIEWER_STDOUT_SELF);
          //   std::cout << "------------------ " << std::endl;
          //   MPI_Comm mpi_comm = MPI_COMM_NULL;
          //   PetscObjectGetComm((PetscObject)rmap, &mpi_comm);
          //   std::cout << "map Comm size: " << MPI::size(mpi_comm) <<
          //   std::endl;
          // }
          // MPI::barrier(MPI_COMM_WORLD);

          // Assemble block
          la::PETScMatrix mat(subA);

          // auto range0 = A.local_range(0);
          // auto range1 = mat.local_range(0);
          // if (MPI::rank(MPI_COMM_WORLD) == 0)
          // {
          //   std::cout << "**R (0): " << range0[0] << ", " << range0[1]
          //             << std::endl;
          //   std::cout << "**R (1): " << range1[0] << ", " << range1[1]
          //             << std::endl;
          // }

          // MPI::barrier(MPI_COMM_WORLD);
          // if (MPI::rank(MPI_COMM_WORLD) == 0)
          // {
          //   std::cout << "Debug data" << std::endl;
          //   auto range0 = mat.local_range(0);
          //   std::cout << "Debug data (a)" << std::endl;
          //   //auto range1 = mat.local_range(1);
          //   std::cout << "Debug data (b)" << std::endl;
          //   std::cout << "Field, range0: " << i << ", " << j << ", "
          //             << range0[0] << ", " << range0[1] << std::endl;
          //   // std::cout << "Range1: " << range1[0] << ", " << range1[1]
          //   //           << std::endl;
          //   std::cout << "End debug data" << std::endl;
          // }

          // MPI::barrier(MPI_COMM_WORLD);

          // std::cout << "Assemble: " << MPI::rank(MPI_COMM_WORLD) <<
          // std::endl;
          // MPI::barrier(MPI_COMM_WORLD);
          // std::cout << "Assemble: " << i << ", " << j << std::endl;
          // MPI::barrier(MPI_COMM_WORLD);
          // MatSetOption(A.mat(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
          //

          // Add bcs to list for diagonal block
          // if (_a[i][j]->function_space(0) == _a[i][j]->function_space(1))
          // {
          //   const PetscInt* l2g;
          //   ISGetIndices(is0, &l2g);

          //   auto space = _a[i][j]->function_space(0);
          //   DirichletBC::Map boundary_values;
          //   for (std::size_t i = 0; i < _bcs.size(); ++i)
          //   {
          //     assert(_bcs[i]);
          //     assert(_bcs[i]->function_space());
          //     if (space->contains(*_bcs[i]->function_space()))
          //     {
          //       // FIXME: find way to avoid gather, or perform with a single
          //       // gather
          //       _bcs[i]->get_boundary_values(boundary_values);
          //       if (MPI::size(MPI_COMM_WORLD) > 1
          //           and _bcs[i]->method() != DirichletBC::Method::pointwise)
          //       {
          //         _bcs[i]->gather(boundary_values);
          //       }
          //     }
          //   }

          //   for (auto bc : boundary_values)
          //   {
          //     la_index_t dof_local = bc.first;
          //     la_index_t dof_global = l2g[dof_local];
          //     bc_values.push_back({dof_global, bc.second});
          //   }

          //   ISRestoreIndices(is0, &l2g);
          // }

          // if (MPI::rank(MPI_COMM_WORLD) == 0)
          // if (i == 1 and j == 1)
          this->assemble(mat, *_a[i][j], _bcs);

          // Restore sub-matrix and destroy index sets
          MatRestoreLocalSubMatrix(A.mat(), is0, is1, &subA);

          ISDestroy(&is0);
          ISDestroy(&is1);

          // A.apply(la::PETScMatrix::AssemblyType::FLUSH);

          offset_col += map1_size;
        }
        else
        {
          // FIXME: Figure out how to check that matrix block is null
          // Null block, do nothing
          throw std::runtime_error("Null block not supported/tested yet.");
        }
      }
      auto map0 = _a[i][0]->function_space(0)->dofmap()->index_map();
      auto map0_size = map0->size(common::IndexMap::MapSize::ALL);
      // auto map0_size = map0->size(common::IndexMap::MapSize::OWNED);
      offset_row += map0_size;
    }

    A.apply(la::PETScMatrix::AssemblyType::FLUSH);

    // // Place '1' on diagonal
    // for (auto bc : bc_values)
    // {
    //   la_index_t row = bc.first;
    //   double one = 1.0;
    //   A.set_local(&one, 1, &row, 1, &row);
    // }
    A.apply(la::PETScMatrix::AssemblyType::FINAL);
  }
  else
  {
    this->assemble(A, *_a[0][0], _bcs);

    A.apply(la::PETScMatrix::AssemblyType::FLUSH);

    // Place '1' on diagonal
    // std::vector<la_index_t> bc_dofs;
    if (_a[0][0]->function_space(0) == _a[0][0]->function_space(1))
    {
      auto space = _a[0][0]->function_space(0);
      DirichletBC::Map boundary_values;
      for (std::size_t i = 0; i < _bcs.size(); ++i)
      {
        assert(_bcs[i]);
        assert(_bcs[i]->function_space());
        if (space->contains(*_bcs[i]->function_space()))
        {
          // FIXME: find way to avoid gather, or perform with a single
          // gather
          _bcs[i]->get_boundary_values(boundary_values);
          if (MPI::size(MPI_COMM_WORLD) > 1
              and _bcs[i]->method() != DirichletBC::Method::pointwise)
          {
            _bcs[i]->gather(boundary_values);
          }
        }
      }

      double one = 1.0;
      for (auto bc : boundary_values)
      {
        PetscInt row = bc.first;
        A.set_local(&one, 1, &row, 1, &row);
      }
    }

    A.apply(la::PETScMatrix::AssemblyType::FINAL);
  }

  std::cout << "Begin assembly" << std::endl;
  A.apply(la::PETScMatrix::AssemblyType::FINAL);
  std::cout << "End  assembly" << std::endl;
}
//-----------------------------------------------------------------------------
void Assembler::assemble(la::PETScVector& b, BlockType block_type)
{
  // Check if matrix should be nested
  assert(!_l.empty());
  const bool block_vector = _l.size() > 1;

  if (b.empty())
  {
    // Initialise matrix if empty

    // Build array of pointers to forms
    std::vector<const Form*> forms(_a.size());
    for (std::size_t i = 0; i < _l.size(); ++i)
      forms[i] = _l[i].get();

    // Initialise vector
    if (block_type == BlockType::nested)
      b = fem::init_nest(forms);
    else if (block_vector and block_type == BlockType::monolithic)
      b = fem::init_monolithic(forms);
    else
      b = fem::init_vector(*_l[0]);
  }

  // auto range = b.local_range();
  // if (MPI::rank(MPI_COMM_WORLD) == 0)
  // {
  //   std::cout << " Local range: " << range[0] << ", " << range[1] <<
  //   std::endl; std::cout << " size: " << b.size() << std::endl;
  // }

  // Get vector type
  VecType vec_type;
  VecGetType(b.vec(), &vec_type);
  bool is_vecnest = strcmp(vec_type, VECNEST) == 0 ? true : false;

  if (is_vecnest)
  {
    for (std::size_t i = 0; i < _l.size(); ++i)
    {
      // Get subvector
      Vec sub_b;
      VecNestGetSubVec(b.vec(), i, &sub_b);
      if (_l[i])
      {
        auto map = _l[i]->function_space(0)->dofmap()->index_map();
        auto map_size = map->size(common::IndexMap::MapSize::ALL);
        auto bs = map->block_size();

        // double* b_array;
        // VecGetArray(sub_b, &b_array);
        // Eigen::Map<EigenVectorXd> _b_array(b_array, map_size);
        // this->assemble(_b_array, *_l[i]);
        // VecRestoreArray(sub_b, &b_array);

        EigenVectorXd _b_array(map_size * bs);
        _b_array.setZero();
        this->assemble(_b_array, *_l[i]);

        la::PETScVector vec(sub_b);
        std::vector<PetscInt> index(map_size * bs);
        std::iota(index.begin(), index.end(), 0);

        vec.add_local(_b_array.data(), map_size * bs, index.data());

        for (std::size_t j = 0; j < _a[i].size(); ++j)
          apply_bc(vec, *_a[i][j], _bcs);
        set_bc(vec, *_l[i], _bcs);

        vec.apply();
      }
      else
      {
        // FIXME: Figure out how to check that vector block is null
        // Null block, do nothing
        std::cout << "WARNING: null linear form. Untested " << std::endl;
      }
    }
  }
  else if (block_vector)
  {
    std::vector<const common::IndexMap*> index_maps;
    for (std::size_t i = 0; i < _l.size(); ++i)
    {
      auto map = _l[i]->function_space(0)->dofmap()->index_map();
      index_maps.push_back(map.get());
    }

    // std::cout << "Assembling block vector (non-nested)" << std::endl;
    std::int64_t offset = 0;
    for (std::size_t i = 0; i < _l.size(); ++i)
    {
      if (_l[i])
      {
        auto map = _l[i]->function_space(0)->dofmap()->index_map();
        auto map_size = map->size(common::IndexMap::MapSize::ALL);

        // Assemble
        EigenVectorXd b_local(map_size);
        b_local.setZero();
        this->assemble(b_local, *_l[i]);

        // Modify vector for bcs
        // for (std::size_t j = 0; j < _a[i].size(); ++j)
        //   apply_bc(b_local, *_a[i][j], _bcs);
        // set_bc(b_local, *_l[i], _bcs);

        // Build local-to-global map
        std::vector<PetscInt> local_to_global_map(
            map->size(common::IndexMap::MapSize::ALL));
        for (std::size_t k = 0; k < local_to_global_map.size(); ++k)
        {
          std::size_t k_global = map->local_to_global(k);
          local_to_global_map[k]
              = fem::get_global_index(index_maps, i, k_global);
        }

        // Add to global vector
        b.add(b_local.data(), map_size, local_to_global_map.data());

        offset += map_size;
      }
    }

    b.apply();
  }
  else
  {
    auto map = _l[0]->function_space(0)->dofmap()->index_map();
    auto map_size = map->size(common::IndexMap::MapSize::ALL);
    auto bs = map->block_size();

    std::cout << "Single Map size: " << map_size << std::endl;
    std::cout << "b size: " << b.size() << std::endl;
    std::cout << "num_maps: " << _l.size() << std::endl;

    // double* b_array;
    // VecGetArray(b.vec(), &b_array);
    // Eigen::Map<EigenVectorXd> _b_array(b_array, map_size);
    EigenVectorXd _b_array(map_size * bs);
    _b_array.setZero();
    this->assemble(_b_array, *_l[0]);

    std::vector<PetscInt> index(map_size * bs);
    std::iota(index.begin(), index.end(), 0);
    b.add_local(_b_array.data(), map_size * bs, index.data());

    // VecRestoreArray(b.vec(), &b_array);

    // this->assemble(b, *_l[0]);
    // apply_bc(b, *_a[0][0], _bcs);
    // set_bc(b, *_l[0], _bcs);
    // b.apply();
    b.apply();
  }
}
//-----------------------------------------------------------------------------
void Assembler::assemble(la::PETScMatrix& A, la::PETScVector& b)
{
  // TODO: pre common boundary condition data

  // Assemble matrix
  assemble(A);

  // Assemble vector
  assemble(b);
}
//-----------------------------------------------------------------------------
void Assembler::assemble(la::PETScMatrix& A, const Form& a,
                         std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  assert(!A.empty());

  // Get mesh from form
  assert(a.mesh());
  const mesh::Mesh& mesh = *a.mesh();

  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim);

  // Function spaces for each axis
  std::array<const function::FunctionSpace*, 2> spaces
      = {{a.function_space(0).get(), a.function_space(1).get()}};

  // Collect pointers to dof maps
  std::array<const GenericDofMap*, 2> dofmaps
      = {{spaces[0]->dofmap().get(), spaces[1]->dofmap().get()}};

  // FIXME: Move out of this function
  // FIXME: For the matrix, we only need to know if there is a boundary
  // condition on the entry. The value is not required.
  // FIXME: Avoid duplication when spaces[0] == spaces[1]
  // Collect boundary conditions by matrix axis
  std::array<DirichletBC::Map, 2> boundary_values;
  for (std::size_t i = 0; i < bcs.size(); ++i)
  {
    assert(bcs[i]);
    assert(bcs[i]->function_space());
    for (std::size_t axis = 0; axis < 2; ++axis)
    {
      if (spaces[axis]->contains(*bcs[i]->function_space()))
      {
        // FIXME: find way to avoid gather, or perform with a single
        // gather
        bcs[i]->get_boundary_values(boundary_values[axis]);
        if (MPI::size(mesh.mpi_comm()) > 1
            and bcs[i]->method() != DirichletBC::Method::pointwise)
        {
          bcs[i]->gather(boundary_values[axis]);
        }
      }
      // else
      // {
      //   std::cout << "     No spaces match: " << std::endl;
      // }
    }
  }

  // Data structures used in assembly
  EigenRowArrayXXd coordinate_dofs;
  EigenRowMatrixXd Ae;

  // Iterate over all cells
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // std::cout << "Iterate over cells" << std::endl;
    // Check that cell is not a ghost
    assert(!cell.is_ghost());

    // Get cell vertex coordinates
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get dof maps for cell
    auto dmap0 = dofmaps[0]->cell_dofs(cell.index());
    auto dmap1 = dofmaps[1]->cell_dofs(cell.index());

    // Size data structure for assembly
    Ae.resize(dmap0.size(), dmap1.size());
    Ae.setZero();

    a.tabulate_tensor(Ae.data(), cell, coordinate_dofs);

    // FIXME: Pass in list  of cells, and list of local dofs, with
    // Dirichlet conditions
    // Note: could use zero dof indices to have PETSc do this
    // Zero rows/columns for Dirichlet bcs
    for (int i = 0; i < Ae.rows(); ++i)
    {
      const std::size_t ii = dmap0[i];
      DirichletBC::Map::const_iterator bc_value = boundary_values[0].find(ii);
      if (bc_value != boundary_values[0].end())
        Ae.row(i).setZero();
    }
    // Loop over columns
    for (int j = 0; j < Ae.cols(); ++j)
    {
      const std::size_t jj = dmap1[j];
      DirichletBC::Map::const_iterator bc_value = boundary_values[1].find(jj);
      if (bc_value != boundary_values[1].end())
        Ae.col(j).setZero();
    }

    // Ae.setZero();

    // Add to matrix
    /*
    std::cout << "Add to matrix: " << std::endl;
    for (std::size_t i = 0; i < dmap0.size(); ++i)
      std::cout << "  0: " << dmap0[i] << std::endl;
    for (std::size_t i = 0; i < dmap1.size(); ++i)
      std::cout << "  1: " << dmap1[i] << std::endl;
  */

    // if (MPI::rank(MPI_COMM_WORLD) == 1)
    {
      // std::cout << "dmaps" << std::endl;
      // std::cout << dmap0 << std::endl;
      // std::cout << "-----" << std::endl;
      // std::cout << dmap1 << std::endl;
      // std::cout << "-----" << std::endl;
      A.add_local(Ae.data(), dmap0.size(), dmap0.data(), dmap1.size(),
                  dmap1.data());
    }
    // std::cout << "Post add to matrix: " << std::endl;
  }

  // Flush matrix
  // A.apply(la::PETScMatrix::AssemblyType::FLUSH);

  // FIXME: Only set if we own the entry
  // FIXME: Move this outside of function? Can then flush later.
  // Place '1' on diagonal for bc entries
  // if (spaces[0] == spaces[1])
  // {
  //   // Note: set diagonal using PETScMatrix::set_local since other functions,
  //   // e.g. PETScMatrix::set_local, do not work for all PETSc Mat types

  //   auto range = A.local_range(0);
  //   std::cout << "**** l comm test: " << MPI::size(A.mpi_comm()) <<
  //   std::endl; if (MPI::rank(MPI_COMM_WORLD) == 0)
  //     std::cout << "**R: " << range[0] << ", " << range[1] << std::endl;
  //   for (auto bc : boundary_values[0])
  //   {
  //     la_index_t row = bc.first;
  //     double one = 1.0;
  //     // if (row >= range[0] and row < range[1])
  //     if (MPI::rank(MPI_COMM_WORLD) > 0)
  //       A.set_local(&one, 1, &row, 1, &row);
  //   }
  // }

  // A.apply(la::PETScMatrix::AssemblyType::FLUSH);

  // Finalise matrix
  // A.apply(la::PETScMatrix::AssemblyType::FINAL);
}
//-----------------------------------------------------------------------------
void Assembler::assemble(Eigen::Ref<EigenVectorXd> b, const Form& L)
{
  // if (b.empty())
  //  init(b, L);

  // Get mesh from form
  assert(L.mesh());
  const mesh::Mesh& mesh = *L.mesh();

  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim);

  // Collect pointers to dof maps
  auto dofmap = L.function_space(0)->dofmap();

  // Data structures used in assembly
  EigenRowArrayXXd coordinate_dofs;
  EigenVectorXd be;

  // Iterate over all cells
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Check that cell is not a ghost
    assert(!cell.is_ghost());

    // Get cell vertex coordinates
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get dof maps for cell
    auto dmap = dofmap->cell_dofs(cell.index());
    // auto dmap1 = dofmaps[1]->cell_dofs(cell.index());

    // Size data structure for assembly
    be.resize(dmap.size());
    be.setZero();

    // Compute cell matrix
    L.tabulate_tensor(be.data(), cell, coordinate_dofs);

    // Add to vector
    for (Eigen::Index i = 0; i < dmap.size(); ++i)
      b[dmap[i]] += be[i];
  }
}
//-----------------------------------------------------------------------------
void Assembler::apply_bc(la::PETScVector& b, const Form& a,
                         std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  // Get mesh from form
  assert(a.mesh());
  const mesh::Mesh& mesh = *a.mesh();

  // Get bcs
  DirichletBC::Map boundary_values;
  for (std::size_t i = 0; i < bcs.size(); ++i)
  {
    assert(bcs[i]);
    assert(bcs[i]->function_space());
    if (a.function_space(1)->contains(*bcs[i]->function_space()))
    {
      bcs[i]->get_boundary_values(boundary_values);
      if (MPI::size(mesh.mpi_comm()) > 1
          and bcs[i]->method() != DirichletBC::Method::pointwise)
      {
        bcs[i]->gather(boundary_values);
      }
    }
  }

  // std::array<const function::FunctionSpace*, 2> spaces
  //    = {{a.function_space(0).get(), a.function_space(1).get()}};

  // Get dofmap for columns a a[i]
  auto dofmap0 = a.function_space(0)->dofmap();
  auto dofmap1 = a.function_space(1)->dofmap();

  EigenRowMatrixXd Ae;
  EigenVectorXd be;
  EigenRowArrayXXd coordinate_dofs;

  // Iterate over all cells
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Check that cell is not a ghost
    assert(!cell.is_ghost());

    // Get dof maps for cell
    auto dmap1 = dofmap1->cell_dofs(cell.index());

    // Check if bc is applied to cell
    bool has_bc = false;
    for (int i = 0; i < dmap1.size(); ++i)
    {
      const std::size_t ii = dmap1[i];
      if (boundary_values.find(ii) != boundary_values.end())
      {
        has_bc = true;
        break;
      }
    }

    // std::cout << "Applying bcs" << std::endl;
    if (!has_bc)
      continue;
    // std::cout << "  has bc" << std::endl;

    // Get cell vertex coordinates
    cell.get_coordinate_dofs(coordinate_dofs);

    // Size data structure for assembly
    auto dmap0 = dofmap1->cell_dofs(cell.index());
    Ae.resize(dmap0.size(), dmap1.size());
    Ae.setZero();
    a.tabulate_tensor(Ae.data(), cell, coordinate_dofs);

    // FIXME: Is this required?
    // Zero Dirichlet rows in Ae
    /*
    if (spaces[0] == spaces[1])
    {
      for (int i = 0; i < dmap0.size(); ++i)
      {
        const std::size_t ii = dmap0[i];
        auto bc = boundary_values.find(ii);
        if (bc != boundary_values.end())
          Ae.row(i).setZero();
      }
    }
    */

    // Size data structure for assembly
    be.resize(dmap0.size());
    be.setZero();

    for (int j = 0; j < dmap1.size(); ++j)
    {
      const std::size_t jj = dmap1[j];
      auto bc = boundary_values.find(jj);
      if (bc != boundary_values.end())
      {
        be -= Ae.col(j) * bc->second;
      }
    }

    // Add to vector
    b.add_local(be.data(), dmap0.size(), dmap0.data());
  }

  // FIXME: Put this elsewhere?
  // Finalise matrix
  b.apply();
}
//-----------------------------------------------------------------------------
void Assembler::set_bc(la::PETScVector& b, const Form& L,
                       std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  // Get mesh from form
  assert(L.mesh());
  const mesh::Mesh& mesh = *L.mesh();

  auto V = L.function_space(0);

  // Get bcs
  DirichletBC::Map boundary_values;
  for (std::size_t i = 0; i < bcs.size(); ++i)
  {
    assert(bcs[i]);
    assert(bcs[i]->function_space());
    if (V->contains(*bcs[i]->function_space()))
    {
      bcs[i]->get_boundary_values(boundary_values);
      if (MPI::size(mesh.mpi_comm()) > 1
          and bcs[i]->method() != DirichletBC::Method::pointwise)
      {
        bcs[i]->gather(boundary_values);
      }
    }
  }

  std::vector<double> values;
  values.reserve(boundary_values.size());
  std::vector<la_index_t> rows;
  rows.reserve(boundary_values.size());
  for (auto bc : boundary_values)
  {
    rows.push_back(bc.first);
    values.push_back(bc.second);
  }

  b.set_local(values.data(), values.size(), rows.data());
  b.apply();
}
//-----------------------------------------------------------------------------
