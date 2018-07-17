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

  // TODO:
  // - check that a is rectangular
  // - a.size() = L.size()
  // - check ranks
  // - check that function spaces in the blocks match, and are not
  //        repeated
  // - figure out number or blocks (row and column)
}
//-----------------------------------------------------------------------------
Assembler::~Assembler()
{
  // for (std::size_t i = 0; i < _block_is.size(); ++i)
  // {
  //   // for (std::size_t j = 0; j < _block_is[j].size(); ++j)
  //   //   if (_block_is[i][j])
  //   //     ISDestroy(&_block_is[i][j]);
  // }
}
//-----------------------------------------------------------------------------
void Assembler::assemble(la::PETScMatrix& A, BlockType block_type)
{
  // Check if matrix should be nested
  assert(!_a.empty());
  const bool block_matrix = _a.size() > 1 or _a[0].size() > 1;

  // Empty bcs (while testing)
  std::vector<std::shared_ptr<const DirichletBC>> bcs = _bcs;

  if (A.empty())
  {
    std::vector<std::vector<const Form*>> forms(
        _a.size(), std::vector<const Form*>(_a[0].size()));
    for (std::size_t i = 0; i < _a.size(); ++i)
      for (std::size_t j = 0; j < _a[i].size(); ++j)
        forms[i][j] = _a[i][j].get();

    // Initialise matrix
    if (block_type == BlockType::nested)
      A = fem::init_nest_matrix(forms);
    else if (block_matrix and block_type == BlockType::monolithic)
      A = fem::init_monolithic_matrix(forms);
    else
      A = fem::init_matrix(*_a[0][0]);
  }

  // Get PETSc matrix type
  MatType mat_type;
  MatGetType(A.mat(), &mat_type);
  const bool is_matnest = strcmp(mat_type, MATNEST) == 0 ? true : false;

  // Collect index sets
  std::vector<std::vector<const common::IndexMap*>> maps(2);
  for (std::size_t i = 0; i < _a.size(); ++i)
    maps[0].push_back(_a[i][0]->function_space(0)->dofmap()->index_map().get());
  for (std::size_t i = 0; i < _a[0].size(); ++i)
    maps[1].push_back(_a[0][i]->function_space(1)->dofmap()->index_map().get());
  std::vector<IS> is_row = compute_index_sets(maps[0]);
  std::vector<IS> is_col = compute_index_sets(maps[1]);

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
    for (std::size_t i = 0; i < _a.size(); ++i)
    {
      for (std::size_t j = 0; j < _a[i].size(); ++j)
      {
        if (_a[i][j])
        {
          if (is_matnest)
          {
            IS is_rows[2], is_cols[2];
            IS is_rowsl[2], is_colsl[2];
            MatNestGetLocalISs(A.mat(), is_rowsl, is_colsl);
            MatNestGetISs(A.mat(), is_rows, is_cols);
            // MPI_Comm mpi_comm = MPI_COMM_NULL;
            // PetscObjectGetComm((PetscObject)is_cols[1], &mpi_comm);
            // std::cout << "Test size: " << MPI::size(mpi_comm) << std::endl;
            if (i == 1 and j == 0)
            {
              if (MPI::rank(MPI_COMM_WORLD) == 1)
              {
                std::cout << "DOLFIN l2g" << std::endl;
                auto index_map
                    = _a[i][j]->function_space(0)->dofmap()->index_map();
                std::cout << "  Range: " << index_map->local_range()[0] << ", "
                          << index_map->local_range()[1] << std::endl;
                std::cout << "Num ghots: " << index_map->num_ghosts()
                          << std::endl;
                auto ghosts = index_map->ghosts();
                std::cout << ghosts << std::endl;
              }

              if (MPI::rank(MPI_COMM_WORLD) == 1)
              {
                ISView(is_rowsl[i], PETSC_VIEWER_STDOUT_SELF);
                ISView(is_row[i], PETSC_VIEWER_STDOUT_SELF);
                std::cout << "----------------------" << std::endl;
              }
              ISView(is_rows[i], PETSC_VIEWER_STDOUT_WORLD);

              Mat subA;
              MatNestGetSubMat(A.mat(), i, j, &subA);
              // std::cout << "Mat (0) address: " << &subA << std::endl;
              la::PETScMatrix Atest(subA);
              // auto orange = Atest.local_range(0);
              // if (MPI::rank(MPI_COMM_WORLD) == 1)
              //   std::cout << "orange: " << orange[0] << ", " << orange[1]
              //             << std::endl;

              ISLocalToGlobalMapping l2g0, l2g1;
              MatGetLocalToGlobalMapping(subA, &l2g0, &l2g1);
              if (MPI::rank(MPI_COMM_WORLD) == 1)
                ISLocalToGlobalMappingView(l2g0, PETSC_VIEWER_STDOUT_SELF);
              // MPI_Comm mpi_comm = MPI_COMM_NULL;
              // PetscObjectGetComm((PetscObject)l2g0, &mpi_comm);
              // std::cout << "Test size: " << MPI::size(mpi_comm) << std::endl;
            }
          }

          Mat subA;
          MatGetLocalSubMatrix(A.mat(), is_row[i], is_col[j], &subA);
          // std::cout << "Mat (1) address: " << &subA << std::endl;

          la::PETScMatrix mat(subA);
          this->assemble(mat, *_a[i][j], bcs);
          if (*_a[i][j]->function_space(0) == *_a[i][j]->function_space(1))
            ident(mat, *_a[i][j]->function_space(0), _bcs);

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
  }
  else
  {
    this->assemble(A, *_a[0][0], bcs);
    if (*_a[0][0]->function_space(0) == *_a[0][0]->function_space(1))
      ident(A, *_a[0][0]->function_space(0), _bcs);
  }

  A.apply(la::PETScMatrix::AssemblyType::FINAL);

  for (std::size_t i = 0; i < is_row.size(); ++i)
    ISDestroy(&is_row[i]);
  for (std::size_t i = 0; i < is_col.size(); ++i)
    ISDestroy(&is_col[i]);
}
//-----------------------------------------------------------------------------
void Assembler::assemble(la::PETScVector& b, BlockType block_type)
{
  // Check if matrix should be nested
  assert(!_l.empty());
  const bool block_vector = _l.size() > 1;

  for (std::size_t i = 0; i < _l.size(); ++i)
  {
    if (!_l[i])
      throw std::runtime_error("Cannot have NULL linear form block.");
  }

  if (b.empty())
  {
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
      b = la::PETScVector(*_l[0]->function_space(0)->dofmap()->index_map());
  }

  // Get vector type
  VecType vec_type;
  VecGetType(b.vec(), &vec_type);
  bool is_vecnest = strcmp(vec_type, VECNEST) == 0 ? true : false;

  if (is_vecnest)
  {
    for (std::size_t i = 0; i < _l.size(); ++i)
    {
      Vec sub_b;
      VecNestGetSubVec(b.vec(), i, &sub_b);
      Vec b_local;
      VecGhostGetLocalForm(sub_b, &b_local);
      assert(b_local);
      this->assemble(b_local, *_l[i]);

      // Modify RHS for Dirichlet bcs
      PetscInt size = 0;
      VecGetSize(b_local, &size);
      PetscScalar* bvalues;
      VecGetArray(b_local, &bvalues);
      Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> bvec(bvalues,
                                                                    size);
      for (std::size_t j = 0; j < _a[i].size(); ++j)
        apply_bc(bvec, *_a[i][j], _bcs);

      VecRestoreArray(b_local, &bvalues);

      VecGhostRestoreLocalForm(sub_b, &b_local);
      VecGhostUpdateBegin(sub_b, ADD_VALUES, SCATTER_REVERSE);
      VecGhostUpdateEnd(sub_b, ADD_VALUES, SCATTER_REVERSE);

      // Set boundary values (local only)
      PetscInt local_size;
      VecGetLocalSize(sub_b, &local_size);
      PetscScalar* values;
      VecGetArray(sub_b, &values);
      Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> vec(values,
                                                                   local_size);
      set_bc(vec, *_l[i], _bcs);
      VecRestoreArray(sub_b, &values);
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
    // Get local representation
    Vec b_local;
    VecGhostGetLocalForm(b.vec(), &b_local);
    assert(b_local);
    PetscScalar* values;
    VecGetArray(b_local, &values);
    for (std::size_t i = 0; i < _l.size(); ++i)
    {
      auto map = _l[i]->function_space(0)->dofmap()->index_map();
      auto map_size0 = map->size_local();
      auto map_size1 = map->num_ghosts();

      int offset0(0), offset1(0);
      for (std::size_t j = 0; j < _l.size(); ++j)
        offset1
            += _l[j]->function_space(0)->dofmap()->index_map()->size_local();
      for (std::size_t j = 0; j < i; ++j)
      {
        offset0
            += _l[j]->function_space(0)->dofmap()->index_map()->size_local();
        offset1
            += _l[j]->function_space(0)->dofmap()->index_map()->num_ghosts();
      }

      // Assemble
      Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> b_vec(map_size0
                                                          + map_size1);
      b_vec.setZero();
      this->assemble(b_vec, *_l[i]);

      // Modify RHS for Dirichlet bcs
      for (std::size_t j = 0; j < _a[i].size(); ++j)
        apply_bc(b_vec, *_a[i][j], _bcs);

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
    for (std::size_t i = 0; i < _l.size(); ++i)
    {
      auto map = _l[i]->function_space(0)->dofmap()->index_map();
      auto map_size0 = map->size_local();

      PetscScalar* values;
      VecGetArray(b.vec(), &values);
      Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> vec(
          values + offset, map_size0);
      set_bc(vec, *_l[i], _bcs);
      VecRestoreArray(b.vec(), &values);
      offset += map_size0;
    }
  }
  else
  {
    // Get local representation
    Vec b_local;
    VecGhostGetLocalForm(b.vec(), &b_local);
    assert(b_local);
    this->assemble(b_local, *_l[0]);

    // Modify RHS for Dirichlet bcs
    PetscInt size = 0;
    VecGetSize(b_local, &size);
    PetscScalar* bvalues;
    VecGetArray(b_local, &bvalues);
    Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> bvec(bvalues,
                                                                  size);
    apply_bc(bvec, *_a[0][0], _bcs);
    VecRestoreArray(b_local, &bvalues);

    // Accumulate ghosts on owning process
    VecGhostRestoreLocalForm(b.vec(), &b_local);

    VecGhostUpdateBegin(b.vec(), ADD_VALUES, SCATTER_REVERSE);
    VecGhostUpdateEnd(b.vec(), ADD_VALUES, SCATTER_REVERSE);

    auto map = _l[0]->function_space(0)->dofmap()->index_map();
    auto map_size0 = map->block_size() * map->size_local();
    PetscScalar* values;
    VecGetArray(b.vec(), &values);
    Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> vec(values,
                                                                 map_size0);
    set_bc(vec, *_l[0], _bcs);
    VecRestoreArray(b.vec(), &values);
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
void Assembler::ident(la::PETScMatrix& A, const function::FunctionSpace& V,
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
  PetscScalar one = 1.0;
  for (auto bc : boundary_values)
  {
    la_index_t row = bc.first;
    if (row < local_size)
      A.add_local(&one, 1, &row, 1, &row);
  }
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
void Assembler::assemble(la::PETScMatrix& A, const Form& a,
                         std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  assert(!A.empty());

  // Get mesh from form
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

  // FIXME: Move out of this function
  // FIXME: For the matrix, we only need to know if there is a boundary
  // condition on the entry. The value is not required.
  // FIXME: Avoid duplication when spaces[0] == spaces[1]
  // Collect boundary conditions by matrix axis
  DirichletBC::Map boundary_values0, boundary_values1;
  for (std::size_t i = 0; i < bcs.size(); ++i)
  {
    assert(bcs[i]);
    assert(bcs[i]->function_space());
    if (V0.contains(*bcs[i]->function_space()))
    {
      // FIXME: find way to avoid gather, or perform with a single
      // gather
      bcs[i]->get_boundary_values(boundary_values0);
      if (MPI::size(mesh.mpi_comm()) > 1
          and bcs[i]->method() != DirichletBC::Method::pointwise)
      {
        bcs[i]->gather(boundary_values0);
      }
    }

    if (V1.contains(*bcs[i]->function_space()))
    {
      bcs[i]->get_boundary_values(boundary_values1);
      if (MPI::size(mesh.mpi_comm()) > 1
          and bcs[i]->method() != DirichletBC::Method::pointwise)
      {
        bcs[i]->gather(boundary_values1);
      }
    }
  }

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
    auto dmap0 = map0.cell_dofs(cell.index());
    auto dmap1 = map1.cell_dofs(cell.index());

    // Size data structure for assembly
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
      DirichletBC::Map::const_iterator bc_value = boundary_values0.find(ii);
      if (bc_value != boundary_values0.end())
        Ae.row(i).setZero();
    }
    // Loop over columns
    for (int j = 0; j < Ae.cols(); ++j)
    {
      const std::size_t jj = dmap1[j];
      DirichletBC::Map::const_iterator bc_value = boundary_values1.find(jj);
      if (bc_value != boundary_values1.end())
        Ae.col(j).setZero();
    }

    A.add_local(Ae.data(), dmap0.size(), dmap0.data(), dmap1.size(),
                dmap1.data());
  }
}
//-----------------------------------------------------------------------------
void Assembler::assemble(Vec b, const Form& L)
{
  // FIXME: Check that we have a sequential vector

  // Get raw array
  PetscScalar* values;
  VecGetArray(b, &values);

  PetscInt size;
  VecGetSize(b, &size);
  Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b_array(values,
                                                                    size);

  assemble(b_array, L);

  VecRestoreArray(b, &values);
}
//-----------------------------------------------------------------------------
void Assembler::assemble(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& L)
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
void Assembler::apply_bc(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b, const Form& a,
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
void Assembler::set_bc(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b, const Form& L,
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

  for (auto bc : boundary_values)
  {
    if (bc.first < (std::size_t)b.size())
      b[bc.first] = bc.second;
  }
}
//-----------------------------------------------------------------------------
