// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <petscis.h>

#include "Assembler.h"
#include "DirichletBC.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "utils.h"
#include <dolfin/common/types.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <string>

using namespace dolfin;
using namespace dolfin::fem;

namespace
{
// Assemble matrix, with Dirichlet rows/columns zeroed. The matrix A
// must already be initialised. The matrix may be a proxy, i.e. a view
// into a larger matrix, and assembly is performed using local
// indices. Matrix is not finalised.
void _assemble_matrix(la::PETScMatrix& A, const Form& a,
                      const std::vector<std::int32_t>& bc_dofs0,
                      const std::vector<std::int32_t>& bc_dofs1)
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
  EigenRowArrayXXd coordinate_dofs;
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
    for (int i = 0; i < Ae.rows(); ++i)
    {
      const std::size_t ii = dmap0[i];
      if (std::find(bc_dofs0.begin(), bc_dofs0.end(), ii) != bc_dofs0.end())
        Ae.row(i).setZero();
    }
    // Loop over columns
    for (int j = 0; j < Ae.cols(); ++j)
    {
      const std::size_t jj = dmap1[j];
      if (std::find(bc_dofs1.begin(), bc_dofs1.end(), jj) != bc_dofs1.end())
        Ae.col(j).setZero();
    }

    A.add_local(Ae.data(), dmap0.size(), dmap0.data(), dmap1.size(),
                dmap1.data());
  }
}

double assemble_scalar(const fem::Form& M)
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
la::PETScVector assemble_vector(const Form& L)
{
  if (L.rank() != 1)
    throw std::runtime_error("Form must be rank 1");
  la::PETScVector b
      = la::PETScVector(*L.function_space(0)->dofmap()->index_map());
  Assembler::assemble_ghosted(b.vec(), L, {}, {});
  return b;
}
//-----------------------------------------------------------------------------
la::PETScMatrix assemble_matrix(const Form& a)
{
  if (a.rank() != 2)
    throw std::runtime_error("Form must be rank 2");
  return fem::assemble({{&a}}, {}, fem::BlockType::monolithic);

  // fem::init_matrix(a);
  // throw std::runtime_error("Short-hand matrix assembly implemented yet.");
  // return A;
}
//-----------------------------------------------------------------------------
void ident(
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
} // namespace

//-----------------------------------------------------------------------------
boost::variant<double, la::PETScVector, la::PETScMatrix>
fem::assemble(const Form& a)
{
  if (a.rank() == 0)
    return assemble_scalar(a);
  else if (a.rank() == 1)
    return assemble_vector(a);
  else if (a.rank() == 2)
    return assemble_matrix(a);
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
              BlockType block_type, double scale)
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

  assemble(b, L, a, bcs, scale);
  return b;
}
//-----------------------------------------------------------------------------
void fem::assemble(
    la::PETScVector& b, std::vector<const Form*> L,
    const std::vector<std::vector<std::shared_ptr<const Form>>> a,
    std::vector<std::shared_ptr<const DirichletBC>> bcs, double scale)
{
  assert(!L.empty());

  VecType vec_type;
  VecGetType(b.vec(), &vec_type);
  bool is_vecnest = strcmp(vec_type, VECNEST) == 0 ? true : false;
  if (is_vecnest)
  {
    for (std::size_t i = 0; i < L.size(); ++i)
    {
      // Get sub-vector and assemble
      Vec sub_b;
      VecNestGetSubVec(b.vec(), i, &sub_b);
      Assembler::assemble_ghosted(sub_b, *L[i], a[i], bcs);
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
      Assembler::assemble_eigen(b_vec, *L[i], a[i], bcs);

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
      set_bc(vec, *L[i], bcs);
      VecRestoreArray(b.vec(), &values);
      offset += map_size0;
    }
  }
  else
    Assembler::assemble_ghosted(b.vec(), *L[0], a[0], bcs);
}
//-----------------------------------------------------------------------------
la::PETScMatrix
fem::assemble(const std::vector<std::vector<const Form*>> a,
              std::vector<std::shared_ptr<const DirichletBC>> bcs,
              BlockType block_type)
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

  assemble(A, a, bcs);
  return A;
}
//-----------------------------------------------------------------------------
void fem::assemble(la::PETScMatrix& A,
                   const std::vector<std::vector<const Form*>> a,
                   std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  // Check if matrix should be nested
  assert(!a.empty());
  const bool block_matrix = a.size() > 1 or a[0].size() > 1;

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
    std::vector<IS> is_row = Assembler::compute_index_sets(maps[0]);
    std::vector<IS> is_col = Assembler::compute_index_sets(maps[1]);

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
          _assemble_matrix(mat, *a[i][j], bc_dofs0, bc_dofs1);
          if (*a[i][j]->function_space(0) == *a[i][j]->function_space(1))
          {
            const Eigen::Array<PetscInt, Eigen::Dynamic, 1> rows
                = Assembler::get_local_bc_rows(*a[i][j]->function_space(0),
                                               bcs);
            ident(mat, rows, 1.0);
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

    _assemble_matrix(A, *a[0][0], bc_dofs0, bc_dofs1);
    if (*a[0][0]->function_space(0) == *a[0][0]->function_space(1))
    {
      const Eigen::Array<PetscInt, Eigen::Dynamic, 1> rows
          = Assembler::get_local_bc_rows(*a[0][0]->function_space(0), bcs);
      ident(A, rows, 1.0);
    }
  }

  A.apply(la::PETScMatrix::AssemblyType::FINAL);
}
//-----------------------------------------------------------------------------
void fem::set_bc(la::PETScVector& b, const Form& L,
                 std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  set_bc(b.vec(), L, bcs);
}
//-----------------------------------------------------------------------------
void fem::set_bc(Vec b, const Form& L,
                 std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  PetscInt local_size;
  VecGetLocalSize(b, &local_size);
  PetscScalar* values;
  VecGetArray(b, &values);
  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> vec(values,
                                                               local_size);
  set_bc(vec, L, bcs);
  VecRestoreArray(b, &values);
}
//-----------------------------------------------------------------------------
void fem::set_bc(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b,
                 const Form& L,
                 std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  // FIXME: optimise this function

  auto V = L.function_space(0);
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> indices;
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> values;
  for (std::size_t i = 0; i < bcs.size(); ++i)
  {
    assert(bcs[i]);
    assert(bcs[i]->function_space());
    if (V->contains(*bcs[i]->function_space()))
    {
      std::tie(indices, values) = bcs[i]->bcs();
      for (Eigen::Index j = 0; j < indices.size(); ++j)
      {
        // FIXME: this check is because DirichletBC::dofs include ghosts
        if (indices[j] < (PetscInt)b.size())
          b[indices[j]] = values[j];
      }
    }
  }
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void Assembler::assemble_ghosted(
    Vec b, const Form& L, const std::vector<std::shared_ptr<const Form>> a,
    const std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  Vec b_local;
  VecGhostGetLocalForm(b, &b_local);
  assemble_local(b_local, L, a, bcs);

  // Restore ghosted form and update local (owned) entries that are
  // ghosts on other processes
  VecGhostRestoreLocalForm(b, &b_local);
  VecGhostUpdateBegin(b, ADD_VALUES, SCATTER_REVERSE);
  VecGhostUpdateEnd(b, ADD_VALUES, SCATTER_REVERSE);

  // Set boundary values (local only)
  set_bc(b, L, bcs);
}
//-----------------------------------------------------------------------------
void Assembler::assemble_local(
    Vec& b, const Form& L, const std::vector<std::shared_ptr<const Form>> a,
    const std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  // FIXME: check that b is a local PETSc Vec

  // Wrap local PETSc Vec as an Eigen vector
  PetscInt size = 0;
  VecGetSize(b, &size);
  PetscScalar* b_array;
  VecGetArray(b, &b_array);
  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> bvec(b_array, size);

  //  Assemble and then modify for Dirichlet bcs  (b  <- b - A x_(bc))
  assemble_eigen(bvec, L, a, bcs);

  // Restore array
  VecRestoreArray(b, &b_array);
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscInt, Eigen::Dynamic, 1> Assembler::get_local_bc_rows(
    const function::FunctionSpace& V,
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
//-----------------------------------------------------------------------------
std::vector<IS>
Assembler::compute_index_sets(std::vector<const common::IndexMap*> maps)
{
  std::vector<IS> is(maps.size());

  std::size_t offset = 0;
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    assert(maps[i]);
    // if (MPI::rank(MPI_COMM_WORLD) == 1)
    //   std::cout << "CCC: " << i << ", " << maps[i]->size_local() << ", "
    //             << maps[i]->num_ghosts() << std::endl;
    const int size = maps[i]->size_local() + maps[i]->num_ghosts();
    std::vector<PetscInt> index(size);
    std::iota(index.begin(), index.end(), offset);
    ISCreateBlock(MPI_COMM_SELF, maps[i]->block_size(), index.size(),
                  index.data(), PETSC_COPY_VALUES, &is[i]);
    offset += size;
  }

  return is;
}
//-----------------------------------------------------------------------------
la::PETScMatrix Assembler::get_sub_matrix(const la::PETScMatrix& A, int i,
                                          int j)
{
  MatType mat_type;
  MatGetType(A.mat(), &mat_type);
  const bool is_matnest = strcmp(mat_type, MATNEST) == 0 ? true : false;

  Mat subA = nullptr;
  if (is_matnest)
    MatNestGetSubMat(A.mat(), i, j, &subA);
  else
  {
    // // Monolithic
    // auto map0 = _a[i][j]->function_space(0)->dofmap()->index_map();
    // auto map1 = _a[i][j]->function_space(1)->dofmap()->index_map();
    // auto map0_size = map0->size_local() + map0->num_ghosts();
    // auto map1_size = map1->size_local() + map1->num_ghosts();
    // std::vector<PetscInt> index0(map0_size), index1(map1_size);
    // std::iota(index0.begin(), index0.end(), offset_row);
    // std::iota(index1.begin(), index1.end(), offset_col);

    // IS is0, is1;
    // ISCreateBlock(MPI_COMM_SELF, map0->block_size(), index0.size(),
    //               index0.data(), PETSC_COPY_VALUES, &is0);
    // ISCreateBlock(MPI_COMM_SELF, map1->block_size(), index1.size(),
    //               index1.data(), PETSC_COPY_VALUES, &is1);

    // MatGetLocalSubMatrix(A.mat(), is0, is1, &subA);
  }

  return la::PETScMatrix(subA);
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void Assembler::assemble_eigen(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& L,
    const std::vector<std::shared_ptr<const Form>> a,
    std::vector<std::shared_ptr<const DirichletBC>> bcs)
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
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;

  // Iterate over all cells
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Check that cell is not a ghost
    assert(!cell.is_ghost());

    // Get cell vertex coordinates
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get dof maps for cell
    auto dmap = dofmap->cell_dofs(cell.index());

    // Size data structure for assembly
    be.resize(dmap.size());
    be.setZero();

    // Compute cell matrix
    L.tabulate_tensor(be.data(), cell, coordinate_dofs);

    // Add to vector
    for (Eigen::Index i = 0; i < dmap.size(); ++i)
      b[dmap[i]] += be[i];
  }

  // Modify for any bcs
  for (std::size_t i = 0; i < a.size(); ++i)
    modify_bc(b, *a[i], bcs);
}
//-----------------------------------------------------------------------------
void Assembler::modify_bc(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b, const Form& a,
    std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  assert(a.rank() == 2);

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

  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;
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

    if (!has_bc)
      continue;

    // Get cell vertex coordinates
    cell.get_coordinate_dofs(coordinate_dofs);

    // Size data structure for assembly
    auto dmap0 = dofmap0->cell_dofs(cell.index());
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

    for (Eigen::Index k = 0; k < dmap0.size(); ++k)
      b[dmap0[k]] += be[k];
  }
}
//-----------------------------------------------------------------------------
