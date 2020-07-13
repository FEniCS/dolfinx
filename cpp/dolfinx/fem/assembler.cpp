// Copyright (C) 2018-2020 Garth N. Wells
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

// namespace
// {
// const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
//                         const std::int32_t*, const PetscScalar*)>
// make_petsc_add(Mat A, [[maybe_unused]] std::vector<PetscInt>& cache)
// {
//   const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
//                           const std::int32_t*, const PetscScalar*)>
//       f = [A, &cache](std::int32_t m, const std::int32_t* rows, std::int32_t
//       n,
//                       const std::int32_t* cols, const PetscScalar* vals) {
//         PetscErrorCode ierr;
// #ifdef PETSC_USE_64BIT_INDICES
//         cache.resize(m + n);
//         std::copy(rows, rows + m, cache.begin());
//         std::copy(cols, cols + n, cache.begin() + m);
//         const PetscInt *rows1 = cache.data(), *cols1 = rows1 + m;
//         ierr = MatSetValuesLocal(A, m, rows1, n, cols1, vals, ADD_VALUES);
// #else
//         cache.data(); // Dummy call to avoid unused variable error
//         ierr = MatSetValuesLocal(A, m, rows, n, cols, vals, ADD_VALUES);
// #endif

// #ifdef DEBUG
//         if (ierr != 0)
//           la::petsc_error(ierr, __FILE__, "MatSetValuesLocal");
// #endif
//         return 0;
//       };
//   return f;
// }
// } // namespace

//-----------------------------------------------------------------------------
void fem::assemble_vector_petsc(Vec b, const Form<PetscScalar>& L)
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
void fem::apply_lifting_petsc(
    Vec b, const std::vector<std::shared_ptr<const Form<PetscScalar>>>& a,
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
void fem::assemble_matrix_petsc(
    Mat A, const Form<PetscScalar>& a,
    const std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>& bcs)
{
  assemble_matrix(la::PETScMatrix::add_fn(A), a, bcs);
}
//-----------------------------------------------------------------------------
void fem::assemble_matrix_petsc(Mat A, const Form<PetscScalar>& a,
                                const std::vector<bool>& bc0,
                                const std::vector<bool>& bc1)
{
  impl::assemble_matrix(la::PETScMatrix::add_fn(A), a, bc0, bc1);
}
//-----------------------------------------------------------------------------
void fem::add_diagonal_petsc(
    Mat A, const function::FunctionSpace& V,
    const std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>& bcs,
    PetscScalar diagonal)
{
  for (const auto& bc : bcs)
  {
    assert(bc);
    if (V.contains(*bc->function_space()))
      add_diagonal_petsc(A, bc->dofs_owned().col(0), diagonal);
  }
}
//-----------------------------------------------------------------------------
void fem::add_diagonal_petsc(
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
void fem::set_bc_petsc(
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

    set_bc<PetscScalar>(_b, bcs, _x0, scale);

    VecRestoreArrayRead(x0_local, &array);
    VecGhostRestoreLocalForm(x0, &x0_local);
  }
  else
    set_bc<PetscScalar>(_b, bcs, scale);

  VecRestoreArray(b, &array);
}
//-----------------------------------------------------------------------------
