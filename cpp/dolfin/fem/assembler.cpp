// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assembler.h"
#include "DirichletBC.h"
#include "DofMap.h"
#include "Form.h"
#include "assemble_matrix_impl.h"
#include "assemble_scalar_impl.h"
#include "assemble_vector_impl.h"
#include "utils.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/types.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>

using namespace dolfin;
using namespace dolfin::fem;

namespace
{
//-----------------------------------------------------------------------------
void set_diagonal_local(
    Mat A,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& rows,
    PetscScalar diag)
{
  assert(A);
  // MatSetOption(A, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE);
  // PetscErrorCode ierr
  //     = MatZeroRowsLocal(A, rows.rows(), rows.data(), diag, nullptr,
  //     nullptr);
  // if (ierr != 0)
  //   la::petsc_error(ierr, __FILE__, "MatZeroRowsLocal");

  PetscInt row_range = 0;
  MatGetLocalSize(A, &row_range, nullptr);

  for (Eigen::Index i = 0; i < rows.size(); ++i)
  {
    const PetscInt row = rows[i];
    MatSetValuesLocal(A, 1, &row, 1, &row, &diag, ADD_VALUES);
  }
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
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& L)
{
  fem::impl::assemble_vector(b, L);
}
//-----------------------------------------------------------------------------
void fem::apply_lifting(
    Vec b, const std::vector<std::shared_ptr<const Form>>& a,
    const std::vector<std::vector<std::shared_ptr<const DirichletBC>>>& bcs1,
    const std::vector<Vec>& x0, double scale)
{
  la::VecWrapper _b(b);
  if (x0.empty())
    fem::impl::apply_lifting(_b.x, a, bcs1, {}, scale);
  else
  {
    std::vector<la::VecReadWrapper> x0_wrapper;
    std::vector<Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>>
        x0_ref;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
      assert(x0[i]);
      x0_wrapper.push_back(la::VecReadWrapper(x0[i]));
      x0_ref.push_back(x0_wrapper.back().x);
    }

    fem::impl::apply_lifting(_b.x, a, bcs1, x0_ref, scale);
  }
}
//-----------------------------------------------------------------------------
void fem::apply_lifting(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const std::vector<std::shared_ptr<const Form>>& a,
    const std::vector<std::vector<std::shared_ptr<const DirichletBC>>>& bcs1,
    const std::vector<
        Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>>& x0,
    double scale)
{
  fem::impl::apply_lifting(b, a, bcs1, x0, scale);
}
//-----------------------------------------------------------------------------
void fem::assemble_matrix_nest(
    Mat A, const std::vector<std::vector<const Form*>>& a,
    const std::vector<std::shared_ptr<const DirichletBC>>& bcs, double diagonal)
{
  assert(A);

  // Loop over each form and assemble
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    for (std::size_t j = 0; j < a[i].size(); ++j)
    {
      if (a[i][j])
      {
        Mat subA;
        MatNestGetSubMat(A, i, j, &subA);
        assemble_matrix_new(subA, *a[i][j], bcs);
        set_diagonal(subA, *a[i][j], bcs, diagonal);
      }
      else
      {
        // Null block, do nothing
      }
    }
  }
}
//-----------------------------------------------------------------------------
void fem::assemble_matrix_block(
    Mat A, const std::vector<std::vector<const Form*>>& a,
    const std::vector<std::shared_ptr<const DirichletBC>>& bcs, double diagonal)
{
  // Check if matrix should be nested
  assert(!a.empty());

  // Prepare data structures for extracting sub-matrices by index sets
  const std::vector<std::vector<std::shared_ptr<const common::IndexMap>>> maps
      = fem::blocked_index_sets(a);
  std::vector<std::vector<const common::IndexMap*>> _maps(2);
  for (auto& m : maps[0])
    _maps[0].push_back(m.get());
  for (auto& m : maps[1])
    _maps[1].push_back(m.get());
  std::vector<IS> is_row = la::compute_petsc_index_sets(_maps[0]);
  std::vector<IS> is_col = la::compute_petsc_index_sets(_maps[1]);

  // Loop over each form and assemble
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    for (std::size_t j = 0; j < a[i].size(); ++j)
    {
      if (a[i][j])
      {
        Mat subA;
        MatGetLocalSubMatrix(A, is_row[i], is_col[j], &subA);
        assemble_matrix_new(subA, *a[i][j], bcs);
        set_diagonal(subA, *a[i][j], bcs, diagonal);
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
}
//-----------------------------------------------------------------------------
void fem::assemble_matrix(
    Mat A, const Form& a,
    const std::vector<std::shared_ptr<const DirichletBC>>& bcs, double diagonal)
{
  // Assemble
  assemble_matrix_new(A, a, bcs);

  // Set diagonal
  set_diagonal(A, a, bcs, diagonal);
}
//-----------------------------------------------------------------------------
void fem::assemble_matrix_new(
    Mat A, const Form& a,
    const std::vector<std::shared_ptr<const DirichletBC>>& bcs)
{
  // Index maps for dof ranges
  auto map0 = a.function_space(0)->dofmap()->index_map;
  auto map1 = a.function_space(1)->dofmap()->index_map;

  // Build dof markers
  std::vector<bool> dof_marker0, dof_marker1;
  std::int32_t dim0
      = map0->block_size * (map0->size_local() + map0->num_ghosts());
  std::int32_t dim1
      = map1->block_size * (map1->size_local() + map1->num_ghosts());
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
}
//-----------------------------------------------------------------------------
void fem::set_diagonal(
    Mat A, const Form& a,
    const std::vector<std::shared_ptr<const DirichletBC>>& bcs, double diagonal)
{
  // Set diagonal for boundary conditions
  auto map0 = a.function_space(0)->dofmap()->index_map;
  if (*a.function_space(0) == *a.function_space(1))
  {
    for (const auto& bc : bcs)
    {
      assert(bc);
      if (a.function_space(0)->contains(*bc->function_space()))
      {
        // FIXME: could be simpler if DirichletBC::dof_indices had
        // options to return owned dofs only
        const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& dofs
            = bc->dof_indices();
        const int owned_size = map0->block_size * map0->size_local();
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
}
//-----------------------------------------------------------------------------
void fem::set_bc(Vec b,
                 const std::vector<std::shared_ptr<const DirichletBC>>& bcs,
                 const Vec x0, double scale)
{
  la::VecWrapper _b(b, false);
  if (x0)
  {
    la::VecReadWrapper _x0(x0, false);
    set_bc(_b.x, bcs, _x0.x, scale);
  }
  else
  {
    set_bc(_b.x, bcs, scale);
  }
}
//-----------------------------------------------------------------------------
void fem::set_bc(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const std::vector<std::shared_ptr<const DirichletBC>>& bcs,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>& x0,
    double scale)
{
  if (b.rows() > x0.rows())
    throw std::runtime_error("Size mismatch between b and x0 vectors.");
  for (auto bc : bcs)
  {
    assert(bc);
    bc->set(b, x0, scale);
  }
}
//-----------------------------------------------------------------------------
void fem::set_bc(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
                 const std::vector<std::shared_ptr<const DirichletBC>>& bcs,
                 double scale)
{
  for (auto bc : bcs)
  {
    assert(bc);
    bc->set(b, scale);
  }
}
//-----------------------------------------------------------------------------
std::vector<std::vector<std::shared_ptr<const fem::DirichletBC>>>
fem::bcs_rows(const std::vector<const Form*>& L,
              const std::vector<std::shared_ptr<const fem::DirichletBC>>& bcs)
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
fem::bcs_cols(const std::vector<std::vector<std::shared_ptr<const Form>>>& a,
              const std::vector<std::shared_ptr<const DirichletBC>>& bcs)
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
