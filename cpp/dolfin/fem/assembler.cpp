// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <petscis.h>

#include "DirichletBC.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "assemble_matrix_impl.h"
#include "assemble_scalar_impl.h"
#include "assemble_vector_impl.h"
#include "assembler.h"
#include "utils.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/types.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>

using namespace dolfin;
using namespace dolfin::fem;

namespace
{
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
std::vector<std::vector<std::shared_ptr<const fem::DirichletBC>>>
bcs_rows(std::vector<const Form*> L,
         std::vector<std::shared_ptr<const fem::DirichletBC>> bcs)
{
  // Pack DirichletBC pointers for rows
  std::vector<std::vector<std::shared_ptr<const fem::DirichletBC>>> bcs0(
      L.size());
  for (std::size_t i = 0; i < L.size(); ++i)
    for (std::shared_ptr<const DirichletBC> bc : bcs)
      if (L[i]->function_space(0)->contains(*bc->function_space()))
        bcs0[i].push_back(bc);

  return bcs0;
}
//-----------------------------------------------------------------------------
std::vector<std::vector<std::vector<std::shared_ptr<const fem::DirichletBC>>>>
bcs_cols(std::vector<std::vector<std::shared_ptr<const Form>>> a,
         std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  // Pack DirichletBC pointers for columns
  std::vector<std::vector<std::vector<std::shared_ptr<const fem::DirichletBC>>>>
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

  return bcs1;
}
//-----------------------------------------------------------------------------
void _assemble_vector_nest(
    Vec b, std::vector<const Form*> L,
    const std::vector<std::vector<std::shared_ptr<const Form>>> a,
    std::vector<std::shared_ptr<const DirichletBC>> bcs, const Vec x0,
    double scale)
{
  if (L.size() < 2)
    throw std::runtime_error("Oops, using blocked assembly.");

  VecType vec_type;
  VecGetType(b, &vec_type);
  const bool is_vecnest = strcmp(vec_type, VECNEST) == 0 ? true : false;
  if (!is_vecnest)
    throw std::runtime_error("Expected a nested vector.");

  // Pack DirichletBC pointers for rows and columns
  std::vector<std::vector<std::shared_ptr<const DirichletBC>>> bcs0
      = bcs_rows(L, bcs);
  std::vector<std::vector<std::vector<std::shared_ptr<const DirichletBC>>>> bcs1
      = bcs_cols(a, bcs);

  for (std::size_t i = 0; i < L.size(); ++i)
  {
    Vec b_sub = nullptr;
    VecNestGetSubVec(b, i, &b_sub);

    // Assemble
    la::VecWrapper _b(b_sub);
    _b.x.setZero();
    fem::impl::assemble_vector(_b.x, *L[i]);

    // FIXME: sort out x0 \ne nullptr for nested case
    // Apply lifting
    if (x0)
    {
      la::VecReadWrapper x0_wrapper(x0);
      std::vector<
          Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>>
          x0_vec(1, x0_wrapper.x);
      fem::impl::apply_lifting(_b.x, a[i], bcs1[i], x0_vec, scale);
      x0_wrapper.restore();
    }
    else
    {
      fem::impl::apply_lifting(_b.x, a[i], bcs1[i], {}, scale);
    }
    _b.restore();

    // Update ghosts
    VecGhostUpdateBegin(b_sub, ADD_VALUES, SCATTER_REVERSE);
    VecGhostUpdateEnd(b_sub, ADD_VALUES, SCATTER_REVERSE);

    // Set bc values
    if (a[0].empty())
    {
      // FIXME: this is a hack to handle the case that no bilinear forms
      // have been supplied, which may happen in a Newton iteration.
      // Needs to be fixed for nested systems
      set_bc(b_sub, bcs0[0], x0, scale);
    }
    else
    {
      for (std::size_t j = 0; j < a[i].size(); ++j)
      {
        if (a[i][j])
        {
          if (*L[i]->function_space(0) == *a[i][j]->function_space(1))
            set_bc(b_sub, bcs0[i], x0, scale);
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void _assemble_vector_block(
    Vec b, std::vector<const Form*> L,
    const std::vector<std::vector<std::shared_ptr<const Form>>> a,
    std::vector<std::shared_ptr<const DirichletBC>> bcs, const Vec x0,
    double scale)
{
  if (L.size() < 2)
    throw std::runtime_error("Oops, using blocked assembly.");

  VecType vec_type;
  VecGetType(b, &vec_type);
  const bool is_vecnest = strcmp(vec_type, VECNEST) == 0 ? true : false;
  if (is_vecnest)
    throw std::runtime_error("Do not expect a nested vector.");

  // const Vec _x0 = x0 ? x0->vec() : nullptr;

  // Pack DirichletBC pointers for rows and columns
  std::vector<std::vector<std::shared_ptr<const DirichletBC>>> bcs0
      = bcs_rows(L, bcs);
  std::vector<std::vector<std::vector<std::shared_ptr<const DirichletBC>>>> bcs1
      = bcs_cols(a, bcs);

  std::vector<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b_vec(L.size());
  for (std::size_t i = 0; i < L.size(); ++i)
  {
    // FIXME: Sort out for x0 \ne nullptr case

    // Get size for block i
    assert(L[i]);
    auto map = L[i]->function_space(0)->dofmap()->index_map();
    const int bs = map->block_size();
    const int map_size0 = map->size_local() * bs;
    const int map_size1 = map->num_ghosts() * bs;
    b_vec[i] = Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>::Zero(map_size0
                                                                   + map_size1);

    // Assemble and modify for bcs (lifting)
    fem::impl::assemble_vector(b_vec[i], *L[i]);
    fem::impl::apply_lifting(b_vec[i], a[i], bcs1[i], {}, scale);
  }

  // Get local representation of b vector and copy values in
  la::VecWrapper _b(b);

  // Compute number of owned (i.e., non-ghost) entries for this process
  int offset1 = 0;
  for (auto& _L : L)
  {
    auto map = _L->function_space(0)->dofmap()->index_map();
    const int bs = map->block_size();
    offset1 += map->size_local() * bs;
  }

  int offset0 = 0;
  for (std::size_t i = 0; i < L.size(); ++i)
  {
    assert(L[i]);
    auto map = L[i]->function_space(0)->dofmap()->index_map();
    const int bs = map->block_size();
    const int map_size0 = map->size_local() * bs;
    const int map_size1 = map->num_ghosts() * bs;

    // Copy data into PETSc b Vec
    Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>& b_i = b_vec[i];
    for (int j = 0; j < map_size0; ++j)
      _b.x[offset0 + j] = b_i[j];
    for (int j = 0; j < map_size1; ++j)
      _b.x[offset1 + j] = b_i[map_size0 + j];

    // Add to local offset for (owned and ghost) for this field
    offset0 += map->size_local() * bs;
    offset1 += map->num_ghosts() * bs;
  }
  _b.restore();

  // FIXME: should this be lifted higher up in the code path?
  // Update ghosts
  VecGhostUpdateBegin(b, ADD_VALUES, SCATTER_REVERSE);
  VecGhostUpdateEnd(b, ADD_VALUES, SCATTER_REVERSE);

  // Set bcs
  PetscScalar* values = nullptr;
  VecGetArray(b, &values);
  int offset = 0;
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
  VecRestoreArray(b, &values);
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
PetscScalar fem::assemble_scalar(const Form& M)
{
  return fem::impl::assemble_scalar(M);
}
//-----------------------------------------------------------------------------
void fem::assemble_vector(Vec b, const Form& L)
{
  la::VecWrapper _b(b);
  fem::impl::assemble_vector(_b.x, L);
}
//-----------------------------------------------------------------------------
void fem::assemble_vector(
    Vec b, std::vector<const Form*> L,
    const std::vector<std::vector<std::shared_ptr<const Form>>> a,
    std::vector<std::shared_ptr<const DirichletBC>> bcs, const Vec x0,
    double scale)
{
  VecType vec_type;
  VecGetType(b, &vec_type);
  const bool is_vecnest = strcmp(vec_type, VECNEST) == 0 ? true : false;
  if (is_vecnest)
    _assemble_vector_nest(b, L, a, bcs, x0, scale);
  else
    _assemble_vector_block(b, L, a, bcs, x0, scale);
}
//-----------------------------------------------------------------------------
void fem::apply_lifting(
    Vec b, const std::vector<std::shared_ptr<const Form>> a,
    std::vector<std::vector<std::shared_ptr<const DirichletBC>>> bcs1,
    const std::vector<Vec> x0, double scale)
{
  if (x0.size() > 1)
  {
    throw std::runtime_error(
        "Simple fem::apply_lifting not get generalised for multiple x0");
  }

  la::VecWrapper _b(b);
  if (x0.empty())
    fem::impl::apply_lifting(_b.x, a, bcs1, {}, scale);
  else
  {
    assert(x0[0]);
    la::VecReadWrapper x0_wrap(x0[0]);
    fem::impl::apply_lifting(_b.x, a, bcs1, {x0_wrap.x}, scale);
    x0_wrap.restore();
  }
  _b.restore();
}
//-----------------------------------------------------------------------------
void fem::assemble_matrix(Mat A, const std::vector<std::vector<const Form*>> a,
                          std::vector<std::shared_ptr<const DirichletBC>> bcs,
                          double diagonal, bool use_nest_extract)
{
  // Check if matrix should be nested
  assert(!a.empty());
  const bool block_matrix = a.size() > 1 or a[0].size() > 1;

  MatType mat_type;
  MatGetType(A, &mat_type);
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
    is_row = la::compute_petsc_index_sets(_maps[0]);
    is_col = la::compute_petsc_index_sets(_maps[1]);
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
          MatGetLocalSubMatrix(A, is_row[i], is_col[j], &subA);
        else if (is_matnest)
          MatNestGetSubMat(A, i, j, &subA);
        else
          subA = A;

        assemble_matrix(subA, *a[i][j], bcs, diagonal);
        if (block_matrix and !is_matnest)
          MatRestoreLocalSubMatrix(A, is_row[i], is_row[j], &subA);
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

  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}
//-----------------------------------------------------------------------------
void fem::assemble_matrix(Mat A, const Form& a,
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
void fem::set_bc(Vec b, std::vector<std::shared_ptr<const DirichletBC>> bcs,
                 const Vec x0, double scale)
{
  la::VecWrapper _b(b, false);
  if (x0)
  {
    la::VecReadWrapper _x0(x0, false);
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
}
//-----------------------------------------------------------------------------
