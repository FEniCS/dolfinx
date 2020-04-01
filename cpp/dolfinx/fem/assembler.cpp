// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
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
#include <Eigen/Sparse>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/mesh/Mesh.h>

using namespace dolfinx;
using namespace dolfinx::fem;

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
      x0_wrapper.emplace_back(x0[i]);
      x0_ref.emplace_back(x0_wrapper.back().x);
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
Eigen::SparseMatrix<PetscScalar, Eigen::RowMajor> fem::assemble_matrix_eigen(
    const Form& a, const std::vector<std::shared_ptr<const DirichletBC>>& bcs)
{
  // Index maps for dof ranges
  auto map0 = a.function_space(0)->dofmap()->index_map;
  auto map1 = a.function_space(1)->dofmap()->index_map;

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

  std::vector<Eigen::Triplet<PetscScalar>> triplets;

  // Lambda function creating Eigen::Triplet array
  const std::function<int(PetscInt, const PetscInt*, PetscInt, const PetscInt*,
                          const PetscScalar*)>
      mat_set_values_local
      = [&triplets](PetscInt nrow, const PetscInt* rows, PetscInt ncol,
                    const PetscInt* cols, const PetscScalar* y) {
          for (int i = 0; i < nrow; ++i)
          {
            int row = rows[i];
            for (int j = 0; j < ncol; ++j)
            {
              int col = cols[j];
              triplets.emplace_back(row, col, y[i * ncol + j]);
            }
          }
          return 0;
        };

  // Assemble
  impl::assemble_matrix(mat_set_values_local, a, dof_marker0, dof_marker1);

  Eigen::SparseMatrix<PetscScalar, Eigen::RowMajor> mat(
      map0->block_size() * (map0->size_local() + map0->num_ghosts()),
      map1->block_size() * (map1->size_local() + map1->num_ghosts()));
  mat.setFromTriplets(triplets.begin(), triplets.end());
  return mat;
}
//-----------------------------------------------------------------------------
void fem::assemble_matrix(
    Mat A, const Form& a,
    const std::vector<std::shared_ptr<const DirichletBC>>& bcs)
{
  // Index maps for dof ranges
  auto map0 = a.function_space(0)->dofmap()->index_map;
  auto map1 = a.function_space(1)->dofmap()->index_map;

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

  const std::function<int(PetscInt, const PetscInt*, PetscInt, const PetscInt*,
                          const PetscScalar*)>
      mat_set_values_local
      = [&A](PetscInt nrow, const PetscInt* rows, PetscInt ncol,
             const PetscInt* cols, const PetscScalar* y) {
          PetscErrorCode ierr
              = MatSetValuesLocal(A, nrow, rows, ncol, cols, y, ADD_VALUES);
#ifdef DEBUG
          if (ierr != 0)
            la::petsc_error(ierr, __FILE__, "MatSetValuesLocal");
#endif
          return 0;
        };

  // Assemble
  impl::assemble_matrix(mat_set_values_local, a, dof_marker0, dof_marker1);
}
//-----------------------------------------------------------------------------
void fem::assemble_matrix(Mat A, const Form& a, const std::vector<bool>& bc0,
                          const std::vector<bool>& bc1)
{
  const std::function<int(PetscInt, const PetscInt*, PetscInt, const PetscInt*,
                          const PetscScalar*)>
      mat_set_values_local
      = [&A](PetscInt nrow, const PetscInt* rows, PetscInt ncol,
             const PetscInt* cols, const PetscScalar* y) {
          PetscErrorCode ierr
              = MatSetValuesLocal(A, nrow, rows, ncol, cols, y, ADD_VALUES);
#ifdef DEBUG
          if (ierr != 0)
            la::petsc_error(ierr, __FILE__, "MatSetValuesLocal");
#endif
          return 0;
        };

  impl::assemble_matrix(mat_set_values_local, a, bc0, bc1);
}
//-----------------------------------------------------------------------------
void fem::add_diagonal(
    Mat A, const function::FunctionSpace& V,
    const std::vector<std::shared_ptr<const DirichletBC>>& bcs,
    PetscScalar diagonal)
{
  for (const auto& bc : bcs)
  {
    assert(bc);
    if (V.contains(*bc->function_space()))
    {
      const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& owned_dofs
          = bc->dofs_owned().col(0);
      add_diagonal(A, owned_dofs, diagonal);
    }
  }
}
//-----------------------------------------------------------------------------
void fem::add_diagonal(
    Mat A,
    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>& rows,
    PetscScalar diagonal)
{
  // NOTE: We use MatSetValuesLocal rather than MatZeroRowsLocal because
  //       MatZeroRowsLocal does not work with sub-matrices extracted
  //       using MatGetLocalSubMatrix from a monolithic matrix.

  // NOTE: MatSetValuesLocal uses ADD_VALUES, hence it requires that the
  //       diagonal is zero before this function is called.

  const std::function<int(PetscInt, const PetscInt*, PetscInt, const PetscInt*,
                          const PetscScalar*)>
      mat_set_values_local
      = [&A](PetscInt nrow, const PetscInt* rows, PetscInt ncol,
             const PetscInt* cols, const PetscScalar* y) {
          PetscErrorCode ierr
              = MatSetValuesLocal(A, nrow, rows, ncol, cols, y, ADD_VALUES);
#ifdef DEBUG
          if (ierr != 0)
            la::petsc_error(ierr, __FILE__, "MatSetValuesLocal");
#endif
          return 0;
        };

  for (Eigen::Index i = 0; i < rows.size(); ++i)
  {
    const PetscInt row = rows(i);
    mat_set_values_local(1, &row, 1, &row, &diagonal);
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
  for (const auto& bc : bcs)
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
  for (const auto& bc : bcs)
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
    for (const std::shared_ptr<const DirichletBC>& bc : bcs)
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
      for (const std::shared_ptr<const DirichletBC>& bc : bcs)
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
