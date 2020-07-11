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

namespace
{
const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                        const std::int32_t*, const PetscScalar*)>
make_petsc_lambda(Mat A, std::vector<PetscInt>& cache)
{
  return [&, A](std::int32_t m, const std::int32_t* rows, std::int32_t n,
                const std::int32_t* cols, const PetscScalar* vals) -> int {
    PetscErrorCode ierr;
    if constexpr (std::is_same<std::int64_t, PetscInt>::value)
    {
      cache.resize(m + n);
      std::copy(rows, rows + m, cache.begin());
      std::copy(cols, cols + n, cache.begin() + m);
      const PetscInt *_rows = cache.data(), *_cols = cache.data() + m;
      ierr = MatSetValuesLocal(A, m, _rows, n, _cols, vals, ADD_VALUES);
    }
    else
    {
      ierr = MatSetValuesLocal(A, m, (PetscInt*)rows, n, (PetscInt*)cols, vals,
                               ADD_VALUES);
    }

#ifdef DEBUG
    if (ierr != 0)
      la::petsc_error(ierr, __FILE__, "MatSetValuesLocal");
#endif
    return 0;
  };
}
} // namespace

//-----------------------------------------------------------------------------
PetscScalar fem::assemble_scalar(const Form& M)
{
  return fem::impl::assemble_scalar<PetscScalar>(M);
}
//-----------------------------------------------------------------------------
void fem::assemble_vector(Vec b, const Form& L)
{
  Vec b_local;
  VecGhostGetLocalForm(b, &b_local);
  PetscInt n = 0;
  VecGetSize(b_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b_local, &array);
  Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> _b(array, n);

  fem::impl::assemble_vector<PetscScalar>(_b, L);

  VecRestoreArray(b_local, &array);
  VecGhostRestoreLocalForm(b, &b_local);
}
//-----------------------------------------------------------------------------
void fem::assemble_vector(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& L)
{
  fem::impl::assemble_vector<PetscScalar>(b, L);
}
//-----------------------------------------------------------------------------
void fem::apply_lifting(
    Vec b, const std::vector<std::shared_ptr<const Form>>& a,
    const std::vector<
        std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>>& bcs1,
    const std::vector<Vec>& x0, double scale)
{
  Vec b_local;
  VecGhostGetLocalForm(b, &b_local);
  PetscInt n = 0;
  VecGetSize(b_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b_local, &array);
  Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> _b(array, n);

  if (x0.empty())
    fem::impl::apply_lifting<PetscScalar>(_b, a, bcs1, {}, scale);
  else
  {
    std::vector<Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>>
        x0_ref;
    std::vector<Vec> x0_local(a.size());
    std::vector<const PetscScalar*> x0_array(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
      assert(x0[i]);
      VecGhostGetLocalForm(x0[i], &x0_local[i]);
      PetscInt n = 0;
      VecGetSize(x0_local[i], &n);
      VecGetArrayRead(x0_local[i], &x0_array[i]);
      x0_ref.emplace_back(x0_array[i], n);
    }

    std::vector<Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>>
        x0_tmp(x0_ref.begin(), x0_ref.end());
    fem::impl::apply_lifting<PetscScalar>(_b, a, bcs1, x0_tmp, scale);

    for (std::size_t i = 0; i < x0_local.size(); ++i)
    {
      VecRestoreArrayRead(x0_local[i], &x0_array[i]);
      VecGhostRestoreLocalForm(x0[i], &x0_local[i]);
    }
  }

  VecRestoreArray(b_local, &array);
  VecGhostRestoreLocalForm(b, &b_local);
}
//-----------------------------------------------------------------------------
void fem::apply_lifting(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const std::vector<std::shared_ptr<const Form>>& a,
    const std::vector<
        std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>>& bcs1,
    const std::vector<
        Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>>& x0,
    double scale)
{
  fem::impl::apply_lifting(b, a, bcs1, x0, scale);
}
//-----------------------------------------------------------------------------
Eigen::SparseMatrix<PetscScalar, Eigen::RowMajor> fem::assemble_matrix_eigen(
    const Form& a,
    const std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>& bcs)
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
  const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                          const std::int32_t*, const PetscScalar*)>
      mat_set_values = [&triplets](std::int32_t nrow, const std::int32_t* rows,
                                   std::int32_t ncol, const std::int32_t* cols,
                                   const PetscScalar* y) {
        for (int i = 0; i < nrow; ++i)
        {
          int row = rows[i];
          for (int j = 0; j < ncol; ++j)
            triplets.emplace_back(row, cols[j], y[i * ncol + j]);
        }
        return 0;
      };

  // Assemble
  impl::assemble_matrix(mat_set_values, a, dof_marker0, dof_marker1);

  Eigen::SparseMatrix<PetscScalar, Eigen::RowMajor> mat(
      map0->block_size() * (map0->size_local() + map0->num_ghosts()),
      map1->block_size() * (map1->size_local() + map1->num_ghosts()));
  mat.setFromTriplets(triplets.begin(), triplets.end());
  return mat;
}
//-----------------------------------------------------------------------------
void fem::assemble_matrix(
    Mat A, const Form& a,
    const std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>& bcs)
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

  std::vector<PetscInt> tmp_dofs_petsc64;
  const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                          const std::int32_t*, const PetscScalar*)>
      mat_set_values = make_petsc_lambda(A, tmp_dofs_petsc64);

  // Assemble
  impl::assemble_matrix(mat_set_values, a, dof_marker0, dof_marker1);
}
//-----------------------------------------------------------------------------
void fem::assemble_matrix(Mat A, const Form& a, const std::vector<bool>& bc0,
                          const std::vector<bool>& bc1)
{
  std::vector<PetscInt> tmp_dofs_petsc64;
  const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                          const std::int32_t*, const PetscScalar*)>
      mat_set_values = make_petsc_lambda(A, tmp_dofs_petsc64);

  impl::assemble_matrix(mat_set_values, a, bc0, bc1);
}
//-----------------------------------------------------------------------------
void fem::add_diagonal(
    Mat A, const function::FunctionSpace& V,
    const std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>& bcs,
    PetscScalar diagonal)
{
  for (const auto& bc : bcs)
  {
    assert(bc);
    if (V.contains(*bc->function_space()))
      add_diagonal(A, bc->dofs_owned().col(0), diagonal);
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

  for (Eigen::Index i = 0; i < rows.size(); ++i)
  {
    const PetscInt row = rows(i);
    PetscErrorCode ierr
        = MatSetValuesLocal(A, 1, &row, 1, &row, &diagonal, ADD_VALUES);
#ifdef DEBUG
    if (ierr != 0)
      la::petsc_error(ierr, __FILE__, "MatSetValuesLocal");
#endif
  }
}
//-----------------------------------------------------------------------------
void fem::set_bc(
    Vec b,
    const std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>& bcs,
    const Vec x0, double scale)
{
  // VecGhostGetLocalForm(b, &b_local);
  PetscInt n = 0;
  VecGetLocalSize(b, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b, &array);
  Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> _b(array, n);

  if (x0)
  {
    Vec x0_local;
    VecGhostGetLocalForm(x0, &x0_local);
    PetscInt n = 0;
    VecGetSize(x0_local, &n);
    const PetscScalar* array = nullptr;
    VecGetArrayRead(x0_local, &array);
    Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> _x0(array,
                                                                        n);

    set_bc(_b, bcs, _x0, scale);

    VecRestoreArrayRead(x0_local, &array);
    VecGhostRestoreLocalForm(x0, &x0_local);
  }
  else
    set_bc(_b, bcs, scale);

  VecRestoreArray(b, &array);
}
//-----------------------------------------------------------------------------
void fem::set_bc(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>& bcs,
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
void fem::set_bc(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>& bcs,
    double scale)
{
  for (const auto& bc : bcs)
  {
    assert(bc);
    bc->set(b, scale);
  }
}
//-----------------------------------------------------------------------------
std::vector<std::vector<std::shared_ptr<const fem::DirichletBC<PetscScalar>>>>
fem::bcs_rows(
    const std::vector<const Form*>& L,
    const std::vector<std::shared_ptr<const fem::DirichletBC<PetscScalar>>>&
        bcs)
{
  // Pack DirichletBC pointers for rows
  std::vector<std::vector<std::shared_ptr<const fem::DirichletBC<PetscScalar>>>>
      bcs0(L.size());
  for (std::size_t i = 0; i < L.size(); ++i)
    for (const std::shared_ptr<const DirichletBC<PetscScalar>>& bc : bcs)
      if (L[i]->function_space(0)->contains(*bc->function_space()))
        bcs0[i].push_back(bc);

  return bcs0;
}
//-----------------------------------------------------------------------------
std::vector<std::vector<
    std::vector<std::shared_ptr<const fem::DirichletBC<PetscScalar>>>>>
fem::bcs_cols(
    const std::vector<std::vector<std::shared_ptr<const Form>>>& a,
    const std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>& bcs)
{
  // Pack DirichletBC pointers for columns
  std::vector<std::vector<
      std::vector<std::shared_ptr<const fem::DirichletBC<PetscScalar>>>>>
      bcs1(a.size());
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    for (std::size_t j = 0; j < a[i].size(); ++j)
    {
      bcs1[i].resize(a[j].size());
      for (const std::shared_ptr<const DirichletBC<PetscScalar>>& bc : bcs)
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
