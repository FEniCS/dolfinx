// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <petscis.h>

#include "DirichletBC.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "assemble_matrix_impl.h"
#include "assemble_vector_impl.h"
#include "assembler.h"
#include "utils.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/types.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>

using namespace dolfin;
using namespace dolfin::fem;

namespace
{
double _assemble_scalar(const fem::Form& M)
{
  if (M.rank() != 0)
    throw std::runtime_error("Form must be rank 0");

  // Get mesh from form
  assert(M.mesh());
  const mesh::Mesh& mesh = *M.mesh();

  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim);

  // Data structure used in assembly
  EigenRowArrayXXd coordinate_dofs;

  // Iterate over all cells
  PetscScalar value = 0.0;
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    PetscScalar cell_value = 0.0;
    assert(!cell.is_ghost());
    cell.get_coordinate_dofs(coordinate_dofs);
    M.tabulate_tensor(&cell_value, cell, coordinate_dofs);
    value += cell_value;
  }

  return MPI::sum(mesh.mpi_comm(), PetscRealPart(value));
}
//-----------------------------------------------------------------------------
la::PETScVector _assemble_vector(const Form& L)
{
  if (L.rank() != 1)
    throw std::runtime_error("Form must be rank 1");
  la::PETScVector b
      = la::PETScVector(*L.function_space(0)->dofmap()->index_map());
  fem::impl::assemble_ghosted(b.vec(), L, {}, {}, nullptr, 1.0);

  return b;
}
//-----------------------------------------------------------------------------
la::PETScMatrix _assemble_matrix(const Form& a)
{
  if (a.rank() != 2)
    throw std::runtime_error("Form must be rank 2");
  return fem::assemble({{&a}}, {}, fem::BlockType::monolithic);
}
//-----------------------------------------------------------------------------
void _ident(
    la::PETScMatrix& A,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> rows,
    PetscScalar diag)
{
  // FIXME: make this process-wise to avoid extra communication step
  // MatZeroRowsLocal(A.mat(), rows.size(), rows.data(), diag, NULL, NULL);
  for (Eigen::Index i = 0; i < rows.size(); ++i)
  {
    const PetscInt row = rows[i];
    A.add_local(&diag, 1, &row, 1, &row);
  }
}
//-----------------------------------------------------------------------------
// Get dof indices that have a boundary condition applied. Indices are
// local and ghost indices are not included.
Eigen::Array<PetscInt, Eigen::Dynamic, 1>
_get_local_bc_rows(const function::FunctionSpace& V,
                   std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  assert(V.mesh());
  const mesh::Mesh& mesh = *V.mesh();

  DirichletBC::Map boundary_values;
  for (std::size_t i = 0; i < bcs.size(); ++i)
  {
    assert(bcs[i]);
    assert(bcs[i]->function_space());
    if (V.contains(*bcs[i]->function_space()))
    {
      // FIXME: find way to avoid gather, or perform with a single
      // gather
      bcs[i]->get_boundary_values(boundary_values);
      if (MPI::size(mesh.mpi_comm()) > 1
          and bcs[i]->method() != DirichletBC::Method::pointwise)
      {
        bcs[i]->gather(boundary_values);
      }
    }
  }

  auto map = V.dofmap()->index_map();
  int local_size = map->block_size() * map->size_local();
  std::vector<PetscInt> _rows;
  for (auto bc : boundary_values)
  {
    PetscInt row = bc.first;
    if (row < local_size)
      _rows.push_back(row);
  }

  Eigen::Array<PetscInt, Eigen::Dynamic, 1> rows
      = Eigen::Map<Eigen::Array<PetscInt, Eigen::Dynamic, 1>>(_rows.data(),
                                                              _rows.size());
  return rows;
}
} // namespace

//-----------------------------------------------------------------------------
boost::variant<double, la::PETScVector, la::PETScMatrix>
fem::assemble(const Form& a)
{
  if (a.rank() == 0)
    return _assemble_scalar(a);
  else if (a.rank() == 1)
    return _assemble_vector(a);
  else if (a.rank() == 2)
    return _assemble_matrix(a);
  else
  {
    throw std::runtime_error("Unsupported rank");
    return 0.0;
  }
}
//-----------------------------------------------------------------------------
la::PETScVector
fem::assemble(std::vector<const Form*> L,
              const std::vector<std::vector<std::shared_ptr<const Form>>> a,
              std::vector<std::shared_ptr<const DirichletBC>> bcs,
              const la::PETScVector* x0, BlockType block_type, double scale)
{
  assert(!L.empty());

  la::PETScVector b;
  const bool block_vector = L.size() > 1;
  if (block_type == BlockType::nested)
    b = fem::init_nest(L);
  else if (block_vector and block_type == BlockType::monolithic)
    b = fem::init_monolithic(L);
  else
    b = la::PETScVector(*L[0]->function_space(0)->dofmap()->index_map());

  assemble(b, L, a, bcs, x0, scale);
  return b;
}
//-----------------------------------------------------------------------------
void fem::assemble(
    la::PETScVector& b, std::vector<const Form*> L,
    const std::vector<std::vector<std::shared_ptr<const Form>>> a,
    std::vector<std::shared_ptr<const DirichletBC>> bcs,
    const la::PETScVector* x0, double scale)
{
  assert(!L.empty());

  VecType vec_type;
  VecGetType(b.vec(), &vec_type);
  bool is_vecnest = strcmp(vec_type, VECNEST) == 0 ? true : false;
  if (is_vecnest)
  {
    // FIXME: Sort out for x0 \ne nullptr case

    for (std::size_t i = 0; i < L.size(); ++i)
    {
      std::vector<std::shared_ptr<const DirichletBC>> _bcs;
      for (std::shared_ptr<const DirichletBC> bc : bcs)
      {
        if (L[i]->function_space(0)->contains(*bc->function_space()))
          _bcs.push_back(bc);
      }

      // Get sub-vector and assemble
      Vec sub_b;
      VecNestGetSubVec(b.vec(), i, &sub_b);
      if (x0)
      {
        fem::impl::assemble_ghosted(sub_b, *L[i], a[i], bcs, x0->vec(), scale);
        fem::impl::set_bc(sub_b, _bcs, x0->vec(), 1.0);
      }
      else
      {
        fem::impl::assemble_ghosted(sub_b, *L[i], a[i], bcs, nullptr, scale);
        fem::impl::set_bc(sub_b, _bcs, 1.0);
      }

      // FIXME: free sub-vector here (dereference)?
    }
  }
  else if (L.size() > 1)
  {
    // FIXME: simplify and hide complexity of this case

    std::vector<const common::IndexMap*> index_maps;
    for (std::size_t i = 0; i < L.size(); ++i)
    {
      auto map = L[i]->function_space(0)->dofmap()->index_map();
      index_maps.push_back(map.get());
    }
    // Get local representation
    Vec b_local;
    VecGhostGetLocalForm(b.vec(), &b_local);
    assert(b_local);
    PetscScalar* values;
    VecGetArray(b_local, &values);
    for (std::size_t i = 0; i < L.size(); ++i)
    {
      auto map = L[i]->function_space(0)->dofmap()->index_map();
      auto map_size0 = map->size_local();
      auto map_size1 = map->num_ghosts();

      int offset0(0), offset1(0);
      for (std::size_t j = 0; j < L.size(); ++j)
        offset1 += L[j]->function_space(0)->dofmap()->index_map()->size_local();
      for (std::size_t j = 0; j < i; ++j)
      {
        offset0 += L[j]->function_space(0)->dofmap()->index_map()->size_local();
        offset1 += L[j]->function_space(0)->dofmap()->index_map()->num_ghosts();
      }

      // Assemble
      Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> b_vec(map_size0
                                                          + map_size1);
      b_vec.setZero();
      fem::impl::assemble_eigen(b_vec, *L[i]);

      // Modify for any essential bcs
      for (std::size_t j = 0; j < a[i].size(); ++j)
        fem::impl::modify_bc(b_vec, *a[i][j], bcs);

      // FIXME: Sort out for x0 \ne nullptr case

      // Copy data into PETSc Vector
      for (int j = 0; j < map_size0; ++j)
        values[offset0 + j] = b_vec[j];
      for (int j = 0; j < map_size1; ++j)
        values[offset1 + j] = b_vec[map_size0 + j];
    }

    VecRestoreArray(b_local, &values);
    VecGhostRestoreLocalForm(b.vec(), &b_local);

    VecGhostUpdateBegin(b.vec(), ADD_VALUES, SCATTER_REVERSE);
    VecGhostUpdateEnd(b.vec(), ADD_VALUES, SCATTER_REVERSE);

    std::size_t offset = 0;
    for (std::size_t i = 0; i < L.size(); ++i)
    {
      auto map = L[i]->function_space(0)->dofmap()->index_map();
      auto map_size0 = map->size_local();

      PetscScalar* values;
      VecGetArray(b.vec(), &values);
      Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> vec(
          values + offset, map_size0);
      Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> vec_x0(nullptr,
                                                                      0);

      std::vector<std::shared_ptr<const DirichletBC>> _bcs;
      for (auto bc : bcs)
      {
        if (L[i]->function_space(0)->contains(*bc->function_space()))
          _bcs.push_back(bc);
      }

      impl::set_bc(vec, _bcs, vec_x0, scale);

      VecRestoreArray(b.vec(), &values);
      offset += map_size0;
    }
  }
  else
  {
    if (x0)
    {
      fem::impl::assemble_ghosted(b.vec(), *L[0], a[0], bcs, x0->vec(), scale);
      impl::set_bc(b.vec(), bcs, x0->vec(), scale);
    }
    else
    {
      fem::impl::assemble_ghosted(b.vec(), *L[0], a[0], bcs, nullptr, scale);
      impl::set_bc(b.vec(), bcs, scale);
    }
  }
}
//-----------------------------------------------------------------------------
la::PETScMatrix
fem::assemble(const std::vector<std::vector<const Form*>> a,
              std::vector<std::shared_ptr<const DirichletBC>> bcs,
              BlockType block_type, double diagonal)
{
  assert(!a.empty());
  const bool block_matrix = a.size() > 1 or a[0].size() > 1;
  la::PETScMatrix A;
  if (block_type == BlockType::nested)
    A = fem::init_nest_matrix(a);
  else if (block_matrix and block_type == BlockType::monolithic)
    A = fem::init_monolithic_matrix(a);
  else
    A = fem::init_matrix(*a[0][0]);

  assemble(A, a, bcs, diagonal);
  return A;
}
//-----------------------------------------------------------------------------
void fem::assemble(la::PETScMatrix& A,
                   const std::vector<std::vector<const Form*>> a,
                   std::vector<std::shared_ptr<const DirichletBC>> bcs,
                   double diagonal)
{
  // Check if matrix should be nested
  assert(!a.empty());
  const bool block_matrix = a.size() > 1 or a[0].size() > 1;

  // FIXME: should zeroing be an option?
  // Zero matrix
  A.zero();

  MatType mat_type;
  MatGetType(A.mat(), &mat_type);
  const bool is_matnest = strcmp(mat_type, MATNEST) == 0 ? true : false;

  // Collect index sets
  std::vector<std::vector<const common::IndexMap*>> maps(2);
  for (std::size_t i = 0; i < a.size(); ++i)
    maps[0].push_back(a[i][0]->function_space(0)->dofmap()->index_map().get());
  for (std::size_t i = 0; i < a[0].size(); ++i)
    maps[1].push_back(a[0][i]->function_space(1)->dofmap()->index_map().get());

  // Assemble matrix
  // if (is_matnest)
  // {
  //   for (std::size_t i = 0; i < _a.size(); ++i)
  //   {
  //     for (std::size_t j = 0; j < _a[i].size(); ++j)
  //     {
  //       if (_a[i][j])
  //       {
  //        la::PETScMatrix Asub = get_sub_matrix(A, i, j);
  //         this->assemble(Asub, *_a[i][j], bcs);
  //         if (*_a[i][j]->function_space(0) == *_a[i][j]->function_space(1))
  //           ident(Asub, *_a[i][j]->function_space(0), _bcs);
  //       }
  //       else
  //       {
  //         throw std::runtime_error("Null block not supported/tested yet.");
  //       }
  //     }
  //   }
  // }
  // else if (block_matrix)
  if (is_matnest or block_matrix)
  {
    std::vector<IS> is_row = la::compute_index_sets(maps[0]);
    std::vector<IS> is_col = la::compute_index_sets(maps[1]);

    for (std::size_t i = 0; i < a.size(); ++i)
    {
      for (std::size_t j = 0; j < a[i].size(); ++j)
      {
        if (a[i][j])
        {
          // if (is_matnest)
          // {
          //   IS is_rows[2], is_cols[2];
          //   IS is_rowsl[2], is_colsl[2];
          //   MatNestGetLocalISs(A.mat(), is_rowsl, is_colsl);
          //   MatNestGetISs(A.mat(), is_rows, is_cols);
          //   // MPI_Comm mpi_comm = MPI_COMM_NULL;
          //   // PetscObjectGetComm((PetscObject)is_cols[1], &mpi_comm);
          //   // std::cout << "Test size: " << MPI::size(mpi_comm) <<
          //   std::endl; if (i == 1 and j == 0)
          //   {
          //     if (MPI::rank(MPI_COMM_WORLD) == 1)
          //     {
          //       std::cout << "DOLFIN l2g" << std::endl;
          //       auto index_map
          //           = _a[i][j]->function_space(0)->dofmap()->index_map();
          //       std::cout << "  Range: " << index_map->local_range()[0] << ",
          //       "
          //                 << index_map->local_range()[1] << std::endl;
          //       auto ghosts = index_map->ghosts();
          //       std::cout << ghosts << std::endl;
          //     }

          //     // if (MPI::rank(MPI_COMM_WORLD) == 1)
          //     // {
          //     //   ISView(is_rowsl[i], PETSC_VIEWER_STDOUT_SELF);
          //     //   ISView(is_row[i], PETSC_VIEWER_STDOUT_SELF);
          //     //   std::cout << "----------------------" << std::endl;
          //     // }
          //     // ISView(is_rows[i], PETSC_VIEWER_STDOUT_WORLD);

          //     Mat subA;
          //     MatNestGetSubMat(A.mat(), i, j, &subA);
          //     // std::cout << "Mat (0) address: " << &subA << std::endl;
          //     // la::PETScMatrix Atest(subA);
          //     // auto orange = Atest.local_range(0);
          //     // if (MPI::rank(MPI_COMM_WORLD) == 1)
          //     //   std::cout << "orange: " << orange[0] << ", " << orange[1]
          //     //             << std::endl;

          //     // ISLocalToGlobalMapping l2g0, l2g1;
          //     // MatGetLocalToGlobalMapping(subA, &l2g0, &l2g1);
          //     // if (MPI::rank(MPI_COMM_WORLD) == 1)
          //     //   ISLocalToGlobalMappingView(l2g0,
          //     PETSC_VIEWER_STDOUT_SELF);
          //     // MPI_Comm mpi_comm = MPI_COMM_NULL;
          //     // PetscObjectGetComm((PetscObject)l2g0, &mpi_comm);
          //     // std::cout << "Test size: " << MPI::size(mpi_comm) <<
          //     std::endl;
          //   }
          // }

          Mat subA;
          MatGetLocalSubMatrix(A.mat(), is_row[i], is_col[j], &subA);
          // std::cout << "Mat (1) address: " << &subA << std::endl;

          std::vector<std::int32_t> bc_dofs0, bc_dofs1;
          for (std::size_t k = 0; k < bcs.size(); ++k)
          {
            assert(bcs[k]);
            assert(bcs[k]->function_space());
            if (a[i][j]->function_space(0)->contains(*bcs[k]->function_space()))
            {
              Eigen::Array<PetscInt, Eigen::Dynamic, 1> bcd
                  = bcs[k]->dof_indices();
              bc_dofs0.insert(bc_dofs0.end(), bcd.data(),
                              bcd.data() + bcd.size());
            }
            if (a[i][j]->function_space(1)->contains(*bcs[k]->function_space()))
            {
              Eigen::Array<PetscInt, Eigen::Dynamic, 1> bcd1
                  = bcs[k]->dof_indices();
              bc_dofs1.insert(bc_dofs1.end(), bcd1.data(),
                              bcd1.data() + bcd1.size());
            }
          }

          la::PETScMatrix mat(subA);
          assemble_matrix(mat, *a[i][j], bc_dofs0, bc_dofs1);
          if (*a[i][j]->function_space(0) == *a[i][j]->function_space(1))
          {
            const Eigen::Array<PetscInt, Eigen::Dynamic, 1> rows
                = _get_local_bc_rows(*a[i][j]->function_space(0), bcs);
            _ident(mat, rows, diagonal);
          }

          MatRestoreLocalSubMatrix(A.mat(), is_row[i], is_row[j], &subA);
        }
        else
        {
          // FIXME: Figure out how to check that matrix block is null
          // Null block, do nothing
          throw std::runtime_error("Null block not supported/tested yet.");
        }
      }
    }
    for (std::size_t i = 0; i < is_row.size(); ++i)
      ISDestroy(&is_row[i]);
    for (std::size_t i = 0; i < is_col.size(); ++i)
      ISDestroy(&is_col[i]);
  }
  else
  {
    std::vector<std::int32_t> bc_dofs0, bc_dofs1;
    for (std::size_t k = 0; k < bcs.size(); ++k)
    {
      assert(bcs[k]);
      assert(bcs[k]->function_space());
      if (a[0][0]->function_space(0)->contains(*bcs[k]->function_space()))
      {
        Eigen::Array<PetscInt, Eigen::Dynamic, 1> bcd0 = bcs[k]->dof_indices();
        bc_dofs0.insert(bc_dofs0.end(), bcd0.data(), bcd0.data() + bcd0.size());
      }
      if (a[0][0]->function_space(1)->contains(*bcs[k]->function_space()))
      {
        Eigen::Array<PetscInt, Eigen::Dynamic, 1> bcd1 = bcs[k]->dof_indices();
        bc_dofs1.insert(bc_dofs1.end(), bcd1.data(), bcd1.data() + bcd1.size());
      }
    }

    assemble_matrix(A, *a[0][0], bc_dofs0, bc_dofs1);
    if (*a[0][0]->function_space(0) == *a[0][0]->function_space(1))
    {
      const Eigen::Array<PetscInt, Eigen::Dynamic, 1> rows
          = _get_local_bc_rows(*a[0][0]->function_space(0), bcs);
      _ident(A, rows, diagonal);
    }
  }

  A.apply(la::PETScMatrix::AssemblyType::FINAL);
}
//-----------------------------------------------------------------------------
void fem::set_bc(la::PETScVector& b,
                 std::vector<std::shared_ptr<const DirichletBC>> bcs,
                 const la::PETScVector* x0, double scale)
{
  if (x0)
    impl::set_bc(b.vec(), bcs, x0->vec(), scale);
  else
    impl::set_bc(b.vec(), bcs, nullptr, scale);
}
//-----------------------------------------------------------------------------
