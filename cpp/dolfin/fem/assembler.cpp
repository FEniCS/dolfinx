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
PetscScalar _assemble_scalar(const fem::Form& M)
{
  if (M.rank() != 0)
    throw std::runtime_error("Form must be rank 0");

  // Get mesh from form
  assert(M.mesh());
  const mesh::Mesh& mesh = *M.mesh();
  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim);

  // Data structure used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs;

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

  return MPI::sum(mesh.mpi_comm(), value);
}
//-----------------------------------------------------------------------------
la::PETScVector _assemble_vector(const Form& L)
{
  if (L.rank() != 1)
    throw std::runtime_error("Form must be rank 1");
  la::PETScVector b
      = la::PETScVector(*L.function_space(0)->dofmap()->index_map());
  la::VecWrapper _b(b.vec());
  _b.x.setZero();
  fem::impl::assemble(_b.x, L);
  _b.restore();

  VecGhostUpdateBegin(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  VecGhostUpdateEnd(b.vec(), ADD_VALUES, SCATTER_REVERSE);

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
void set_diagonal_local(
    Mat A,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> rows,
    PetscScalar diag)
{
  assert(A);
  for (Eigen::Index i = 0; i < rows.size(); ++i)
  {
    const PetscInt row = rows[i];
    PetscErrorCode ierr
        = MatSetValuesLocal(A, 1, &row, 1, &row, &diag, ADD_VALUES);
    if (ierr != 0)
      la::petsc_error(ierr, __FILE__, "MatSetValuesLocal");
  }
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
boost::variant<PetscScalar, la::PETScVector, la::PETScMatrix>
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
  const Vec _x0 = x0 ? x0->vec() : nullptr;

  // Packs DirichletBC pointers for rows
  std::vector<std::vector<std::shared_ptr<const DirichletBC>>> bcs0(L.size());
  for (std::size_t i = 0; i < L.size(); ++i)
    for (std::shared_ptr<const DirichletBC> bc : bcs)
      if (L[i]->function_space(0)->contains(*bc->function_space()))
        bcs0[i].push_back(bc);

  // Packs DirichletBC pointers for columns
  std::vector<std::vector<std::vector<std::shared_ptr<const DirichletBC>>>>
      bcs1(a.size());
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    for (std::size_t j = 0; j < a[i].size(); ++j)
    {
      bcs1[i].resize(a[j].size());
      for (std::shared_ptr<const DirichletBC> bc : bcs)
      {
        // FIXME: handle case where a[i][j] is null
        if (a[i][j])
        {
          if (a[i][j]->function_space(1)->contains(*bc->function_space()))
            bcs1[i][j].push_back(bc);
        }
      }
    }
  }

  VecType vec_type;
  VecGetType(b.vec(), &vec_type);
  bool is_vecnest = strcmp(vec_type, VECNEST) == 0 ? true : false;
  if (L.size() == 1 or is_vecnest)
  {
    // FIXME: Sort out for x0 \ne nullptr case

    for (std::size_t i = 0; i < L.size(); ++i)
    {
      // FIXME: need to extract block of x0
      Vec b_sub = b.vec();
      if (is_vecnest)
        VecNestGetSubVec(b.vec(), i, &b_sub);

      // Assemble
      assemble_petsc(b_sub, *L[i], a[i], bcs1[i], _x0, scale);
      VecGhostUpdateBegin(b_sub, ADD_VALUES, SCATTER_REVERSE);
      VecGhostUpdateEnd(b_sub, ADD_VALUES, SCATTER_REVERSE);

      // Set bc values
      if (a[0].empty())
      {
        // FIXME: this is a hack to handle the case that no bilinear
        // forms have been supplied, which may happen in a Newton
        // iteration. Needs to be fixed for nested systems
        set_bc_petsc(b_sub, bcs0[0], _x0, scale);
      }
      else
      {
        for (std::size_t j = 0; j < a[i].size(); ++j)
        {
          if (a[i][j])
          {
            if (*L[i]->function_space(0) == *a[i][j]->function_space(1))
              set_bc_petsc(b_sub, bcs0[i], _x0, scale);
          }
        }
      }
    }
  }
  else if (L.size() > 1)
  {
    // Get local representation of b vector and unwrap
    la::VecWrapper _b(b.vec());
    for (std::size_t i = 0; i < L.size(); ++i)
    {
      // FIXME: Sort out for x0 \ne nullptr case

      // Get size for block i
      assert(L[i]);
      auto map = L[i]->function_space(0)->dofmap()->index_map();
      const int bs = map->block_size();
      const int map_size0 = map->size_local() * bs;
      const int map_size1 = map->num_ghosts() * bs;

      // Assemble and modify for bcs (lifting)
      Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> b_vec
          = Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>::Zero(map_size0
                                                                + map_size1);
      assemble_eigen(b_vec, *L[i], a[i], bcs1[i], {}, scale);

      // Compute offsets for block i
      int offset0(0), offset1(0);
      for (auto& _L : L)
      {
        auto map = _L->function_space(0)->dofmap()->index_map();
        offset1 += map->size_local() * map->block_size();
      }
      for (std::size_t j = 0; j < i; ++j)
      {
        auto map = L[j]->function_space(0)->dofmap()->index_map();
        const int bs = map->block_size();
        offset0 += map->size_local() * bs;
        offset1 += map->num_ghosts() * bs;
      }

      // Copy data into PETSc b Vec
      for (int j = 0; j < map_size0; ++j)
        _b.x[offset0 + j] = b_vec[j];
      for (int j = 0; j < map_size1; ++j)
        _b.x[offset1 + j] = b_vec[map_size0 + j];
    }
    _b.restore();

    // FIXME: should this be lifted higher up in the code path?
    // Update ghosts
    VecGhostUpdateBegin(b.vec(), ADD_VALUES, SCATTER_REVERSE);
    VecGhostUpdateEnd(b.vec(), ADD_VALUES, SCATTER_REVERSE);

    // Set bcs
    PetscScalar* values;
    VecGetArray(b.vec(), &values);
    std::size_t offset = 0;
    for (std::size_t i = 0; i < L.size(); ++i)
    {
      auto map = L[i]->function_space(0)->dofmap()->index_map();
      const int bs = map->block_size();
      const int map_size0 = map->size_local() * bs;
      Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> vec(
          values + offset, map_size0);
      Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> vec_x0(nullptr,
                                                                       0);
      for (auto bc : bcs)
      {
        if (L[i]->function_space(0)->contains(*bc->function_space()))
          bc->set(vec, scale);
      }
      offset += map_size0;
    }
    VecRestoreArray(b.vec(), &values);
  }
}
//-----------------------------------------------------------------------------
void fem::assemble_petsc(
    Vec b, const Form& L, const std::vector<std::shared_ptr<const Form>> a,
    std::vector<std::vector<std::shared_ptr<const DirichletBC>>> bcs1,
    const Vec x0, double scale)
{
  la::VecWrapper _b(b);
    _b.x.setZero();
  if (x0)
  {
    std::vector<la::VecReadWrapper> _x0;
    if (a.size() > 1)
    {
      // FIXME: Add some checks
      throw std::runtime_error("Not implemented yet.");
      PetscInt n = 1;
      Vec* _x0_sub = nullptr;
      VecNestGetSubVecs(x0, &n, &_x0_sub);
      for (PetscInt i = 0; i < n; ++i)
        _x0.push_back(la::VecReadWrapper(_x0_sub[i]));
    }
    else
      _x0.push_back(la::VecReadWrapper(x0));

    // Wrap the x0 vector(s)
    std::vector<Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>>
        _x0_ref;
    for (std::size_t j = 0; j < _x0.size(); ++j)
      _x0_ref.push_back(_x0[j].x);

    // Assemble and modify for bcs
    assemble_eigen(_b.x, L, a, bcs1, _x0_ref, scale);

    // Restore the x0 vectors
    for (auto& x : _x0)
      x.restore();
  }
  else
    assemble_eigen(_b.x, L, a, bcs1, {}, scale);

  _b.restore();
}
//-----------------------------------------------------------------------------
void fem::assemble_eigen(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& L,
    const std::vector<std::shared_ptr<const Form>> a,
    std::vector<std::vector<std::shared_ptr<const DirichletBC>>> bcs1,
    std::vector<Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>>
        x0,
    double scale)
{
  // FIXME: make changes to reactivate this check
  // if (!x0.empty() and x0.size() != a.size())
  //   throw std::runtime_error("Mismatch in size between x0 and a in
  //   assembler.");
  if (a.size() != bcs1.size())
  {
    throw std::runtime_error(
        "Mismatch in size between a and bcs in assembler.");
  }

  // Assemble b
  fem::impl::assemble(b, L);

  // Modify for Dirichlet bcs (lifting)
  for (std::size_t j = 0; j < a.size(); ++j)
  {
    std::vector<bool> bc_markers1;
    Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> bc_values1;
    if (a[j] and !bcs1[j].empty())
    {
      auto V1 = a[j]->function_space(1);
      assert(V1);
      auto map1 = V1->dofmap()->index_map();
      assert(map1);
      const int crange
          = map1->block_size() * (map1->size_local() + map1->num_ghosts());
      bc_markers1.assign(crange, false);
      bc_values1 = Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>::Zero(crange);
      for (std::shared_ptr<const DirichletBC>& bc : bcs1[j])
      {
        bc->mark_dofs(bc_markers1);
        bc->dof_values(bc_values1);
      }

      // Modify (apply lifting) vector
      if (!x0.empty())
        fem::impl::lift_bc(b, *a[j], bc_values1, bc_markers1, x0[j], scale);
      else
        fem::impl::lift_bc(b, *a[j], bc_values1, bc_markers1, scale);
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
                   double diagonal, bool use_nest_extract)
{
  // Check if matrix should be nested
  assert(!a.empty());
  const bool block_matrix = a.size() > 1 or a[0].size() > 1;

  // FIXME: should zeroing be an option?
  // Zero matrix
  A.zero();

  MatType mat_type;
  MatGetType(A.mat(), &mat_type);
  const bool is_matnest
      = (strcmp(mat_type, MATNEST) == 0) and use_nest_extract ? true : false;

  // Assemble matrix
  std::vector<IS> is_row, is_col;
  if (block_matrix and !is_matnest)
  {
    // Prepare data structures for extracting sub-matrices by index
    // sets

    // Extract index maps
    const std::vector<std::vector<std::shared_ptr<const common::IndexMap>>> maps
        = fem::blocked_index_sets(a);
    std::vector<std::vector<const common::IndexMap*>> _maps(2);
    for (auto& m : maps[0])
      _maps[0].push_back(m.get());
    for (auto& m : maps[1])
      _maps[1].push_back(m.get());
    is_row = la::compute_index_sets(_maps[0]);
    is_col = la::compute_index_sets(_maps[1]);
  }

  // Loop over each form and assemble
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    for (std::size_t j = 0; j < a[i].size(); ++j)
    {
      if (a[i][j])
      {
        Mat subA;
        if (block_matrix and !is_matnest)
          MatGetLocalSubMatrix(A.mat(), is_row[i], is_col[j], &subA);
        else if (is_matnest)
          MatNestGetSubMat(A.mat(), i, j, &subA);
        else
          subA = A.mat();

        assemble_petsc(subA, *a[i][j], bcs, diagonal);
        if (block_matrix and !is_matnest)
          MatRestoreLocalSubMatrix(A.mat(), is_row[i], is_row[j], &subA);
      }
      else
      {
        // Null block, do nothing
      }
    }
  }

  // Clean up index sets
  for (std::size_t i = 0; i < is_row.size(); ++i)
    ISDestroy(&is_row[i]);
  for (std::size_t i = 0; i < is_col.size(); ++i)
    ISDestroy(&is_col[i]);

  A.apply(la::PETScMatrix::AssemblyType::FINAL);
}
//-----------------------------------------------------------------------------
void fem::assemble_petsc(Mat A, const Form& a,
                         std::vector<std::shared_ptr<const DirichletBC>> bcs,
                         double diagonal)
{
  // Index maps for dof ranges
  auto map0 = a.function_space(0)->dofmap()->index_map();
  auto map1 = a.function_space(1)->dofmap()->index_map();

  // Build dof markers
  std::vector<bool> dof_marker0, dof_marker1;
  std::int32_t dim0
      = map0->block_size() * (map0->size_local() + map0->num_ghosts());
  std::int32_t dim1
      = map1->block_size() * (map1->size_local() + map1->num_ghosts());
  for (std::size_t k = 0; k < bcs.size(); ++k)
  {
    assert(bcs[k]);
    assert(bcs[k]->function_space());
    if (a.function_space(0)->contains(*bcs[k]->function_space()))
    {
      dof_marker0.resize(dim0, false);
      bcs[k]->mark_dofs(dof_marker0);
    }
    if (a.function_space(1)->contains(*bcs[k]->function_space()))
    {
      dof_marker1.resize(dim1, false);
      bcs[k]->mark_dofs(dof_marker1);
    }
  }

  // Assemble
  impl::assemble_matrix(A, a, dof_marker0, dof_marker1);

  // Set diagonal for boundary conditions
  if (*a.function_space(0) == *a.function_space(1))
  {
    for (const auto& bc : bcs)
    {
      assert(bc);
      if (a.function_space(0)->contains(*bc->function_space()))
      {
        // FIXME: could be simpler if DirichletBC::dof_indices had
        // options to return owned dofs only
        const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dofs
            = bc->dof_indices();
        const int owned_size = map0->block_size() * map0->size_local();
        auto it = std::lower_bound(dofs.data(), dofs.data() + dofs.rows(),
                                   owned_size);
        const Eigen::Index pos = std::distance(dofs.data(), it);
        assert(pos <= dofs.size() and pos >= 0);
        const Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
            dofs_owned(dofs.data(), pos);
        set_diagonal_local(A, dofs_owned, diagonal);
      }
    }
  }

  // Do not finalise assembly - matrix may be a proxy/sub-matrix with
  // finalisation done elsewhere.
}
//-----------------------------------------------------------------------------
void fem::set_bc(la::PETScVector& b,
                 std::vector<std::shared_ptr<const DirichletBC>> bcs,
                 const la::PETScVector* x0, double scale)
{
  const Vec _x0 = (x0 != nullptr) ? x0->vec() : nullptr;
  set_bc_petsc(b.vec(), bcs, _x0, scale);
}
//-----------------------------------------------------------------------------
void fem::set_bc_petsc(Vec b,
                       std::vector<std::shared_ptr<const DirichletBC>> bcs,
                       const Vec x0, double scale)
{
  la::VecWrapper _b(b);
  if (x0)
  {
    la::VecReadWrapper _x0(x0);
    if (_b.x.size() != _x0.x.size())
      throw std::runtime_error("Size mismatch between b and x0 vectors.");
    for (auto bc : bcs)
    {
      assert(bc);
      bc->set(_b.x, _x0.x, scale);
    }
  }
  else
  {
    for (auto bc : bcs)
    {
      assert(bc);
      bc->set(_b.x, scale);
    }
  }

  _b.restore();
}
//-----------------------------------------------------------------------------
