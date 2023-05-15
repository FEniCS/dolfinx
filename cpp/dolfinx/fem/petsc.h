// Copyright (C) 2018-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Form.h"
#include "assembler.h"
#include "utils.h"
#include <concepts>
#include <dolfinx/la/petsc.h>
#include <map>
#include <memory>
#include <petscmat.h>
#include <petscvec.h>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::fem
{
template <typename T, std::floating_point U>
class DirichletBC;

/// @brief Helper functions for assembly into PETSc data structures
namespace petsc
{
/// Create a matrix
/// @param[in] a A bilinear form
/// @param[in] type The PETSc matrix type to create
/// @return A sparse matrix with a layout and sparsity that matches the
/// bilinear form. The caller is responsible for destroying the Mat
/// object.
template <std::floating_point T>
Mat create_matrix(const Form<PetscScalar, T>& a,
                  const std::string& type = std::string())
{
  la::SparsityPattern pattern = fem::create_sparsity_pattern(a);
  pattern.finalize();
  return la::petsc::create_matrix(a.mesh()->comm(), pattern, type);
}

/// Initialise a monolithic matrix for an array of bilinear forms
/// @param[in] a Rectangular array of bilinear forms. The `a(i, j)` form
/// will correspond to the `(i, j)` block in the returned matrix
/// @param[in] type The type of PETSc Mat. If empty the PETSc default is
/// used.
/// @return A sparse matrix  with a layout and sparsity that matches the
/// bilinear forms. The caller is responsible for destroying the Mat
/// object.
template <std::floating_point T>
Mat create_matrix_block(
    const std::vector<std::vector<const Form<PetscScalar, T>*>>& a,
    const std::string& type = std::string())
{
  // Extract and check row/column ranges
  std::array<std::vector<std::shared_ptr<const FunctionSpace<T>>>, 2> V
      = fem::common_function_spaces(extract_function_spaces(a));
  std::array<std::vector<int>, 2> bs_dofs;
  for (std::size_t i = 0; i < 2; ++i)
  {
    for (auto& _V : V[i])
      bs_dofs[i].push_back(_V->dofmap()->bs());
  }

  // Build sparsity pattern for each block
  std::shared_ptr<const mesh::Mesh<T>> mesh;
  std::vector<std::vector<std::unique_ptr<la::SparsityPattern>>> patterns(
      V[0].size());
  for (std::size_t row = 0; row < V[0].size(); ++row)
  {
    for (std::size_t col = 0; col < V[1].size(); ++col)
    {
      if (const Form<PetscScalar, T>* form = a[row][col]; form)
      {
        patterns[row].push_back(std::make_unique<la::SparsityPattern>(
            create_sparsity_pattern(*form)));
        if (!mesh)
          mesh = form->mesh();
      }
      else
        patterns[row].push_back(nullptr);
    }
  }

  if (!mesh)
    throw std::runtime_error("Could not find a Mesh.");

  // Compute offsets for the fields
  std::array<std::vector<std::pair<
                 std::reference_wrapper<const common::IndexMap>, int>>,
             2>
      maps;
  for (std::size_t d = 0; d < 2; ++d)
  {
    for (auto space : V[d])
    {
      maps[d].emplace_back(*space->dofmap()->index_map,
                           space->dofmap()->index_map_bs());
    }
  }

  // Create merged sparsity pattern
  std::vector<std::vector<const la::SparsityPattern*>> p(V[0].size());
  for (std::size_t row = 0; row < V[0].size(); ++row)
    for (std::size_t col = 0; col < V[1].size(); ++col)
      p[row].push_back(patterns[row][col].get());

  la::SparsityPattern pattern(mesh->comm(), p, maps, bs_dofs);
  pattern.finalize();

  // FIXME: Add option to pass customised local-to-global map to PETSc
  // Mat constructor

  // Initialise matrix
  Mat A = la::petsc::create_matrix(mesh->comm(), pattern, type);

  // Create row and column local-to-global maps (field0, field1, field2,
  // etc), i.e. ghosts of field0 appear before owned indices of field1
  std::array<std::vector<PetscInt>, 2> _maps;
  for (int d = 0; d < 2; ++d)
  {
    // FIXME: Index map concatenation has already been computed inside
    // the SparsityPattern constructor, but we also need it here to
    // build the PETSc local-to-global map. Compute outside and pass
    // into SparsityPattern constructor.

    // FIXME: avoid concatenating the same maps twice in case that V[0]
    // == V[1].

    // Concatenate the block index map in the row and column directions
    auto [rank_offset, local_offset, ghosts, _]
        = common::stack_index_maps(maps[d]);
    for (std::size_t f = 0; f < maps[d].size(); ++f)
    {
      const common::IndexMap& map = maps[d][f].first.get();
      const int bs = maps[d][f].second;
      const std::int32_t size_local = bs * map.size_local();
      const std::vector global = map.global_indices();
      for (std::int32_t i = 0; i < size_local; ++i)
        _maps[d].push_back(i + rank_offset + local_offset[f]);
      for (std::size_t i = size_local; i < bs * global.size(); ++i)
        _maps[d].push_back(ghosts[f][i - size_local]);
    }
  }

  // Create PETSc local-to-global map/index sets and attach to matrix
  ISLocalToGlobalMapping petsc_local_to_global0;
  ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 1, _maps[0].size(),
                               _maps[0].data(), PETSC_COPY_VALUES,
                               &petsc_local_to_global0);
  if (V[0] == V[1])
  {
    MatSetLocalToGlobalMapping(A, petsc_local_to_global0,
                               petsc_local_to_global0);
    ISLocalToGlobalMappingDestroy(&petsc_local_to_global0);
  }
  else
  {

    ISLocalToGlobalMapping petsc_local_to_global1;
    ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 1, _maps[1].size(),
                                 _maps[1].data(), PETSC_COPY_VALUES,
                                 &petsc_local_to_global1);
    MatSetLocalToGlobalMapping(A, petsc_local_to_global0,
                               petsc_local_to_global1);
    ISLocalToGlobalMappingDestroy(&petsc_local_to_global0);
    ISLocalToGlobalMappingDestroy(&petsc_local_to_global1);
  }

  return A;
}

/// @brief Create nested (MatNest) matrix.
///
/// @note The caller is responsible for destroying the Mat object.
template <std::floating_point T>
Mat create_matrix_nest(
    const std::vector<std::vector<const Form<PetscScalar, T>*>>& a,
    const std::vector<std::vector<std::string>>& types)
{
  // Extract and check row/column ranges
  auto V = fem::common_function_spaces(extract_function_spaces(a));

  std::vector<std::vector<std::string>> _types(
      a.size(), std::vector<std::string>(a.front().size()));
  if (!types.empty())
    _types = types;

  // Loop over each form and create matrix
  int rows = a.size();
  int cols = a.front().size();
  std::vector<Mat> mats(rows * cols, nullptr);
  std::shared_ptr<const mesh::Mesh<T>> mesh;
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      if (const Form<PetscScalar, T>* form = a[i][j]; form)
      {
        mats[i * cols + j] = create_matrix(*form, _types[i][j]);
        mesh = form->mesh();
      }
    }
  }

  if (!mesh)
    throw std::runtime_error("Could not find a Mesh.");

  // Initialise block (MatNest) matrix
  Mat A;
  MatCreate(mesh->comm(), &A);
  MatSetType(A, MATNEST);
  MatNestSetSubMats(A, rows, nullptr, cols, nullptr, mats.data());
  MatSetUp(A);

  // De-reference Mat objects
  for (std::size_t i = 0; i < mats.size(); ++i)
  {
    if (mats[i])
      MatDestroy(&mats[i]);
  }

  return A;
}

/// Initialise monolithic vector. Vector is not zeroed.
///
/// The caller is responsible for destroying the Mat object
Vec create_vector_block(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps);

/// Create nested (VecNest) vector. Vector is not zeroed.
Vec create_vector_nest(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps);

// -- Vectors ----------------------------------------------------------------

/// @brief Assemble linear form into an already allocated PETSc vector.
///
/// Ghost contributions are not accumulated (not sent to owner). Caller
/// is responsible for calling `VecGhostUpdateBegin/End`.
///
/// @param[in,out] b The PETsc vector to assemble the form into. The
/// vector must already be initialised with the correct size. The
/// process-local contribution of the form is assembled into this
/// vector. It is not zeroed before assembly.
/// @param[in] L The linear form to assemble
/// @param[in] constants The constants that appear in `L`
/// @param[in] coeffs The coefficients that appear in `L`
template <std::floating_point T>
void assemble_vector(
    Vec b, const Form<PetscScalar, T>& L,
    std::span<const PetscScalar> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const PetscScalar>, int>>& coeffs)
{
  Vec b_local;
  VecGhostGetLocalForm(b, &b_local);
  PetscInt n = 0;
  VecGetSize(b_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b_local, &array);
  std::span<PetscScalar> _b(array, n);
  fem::assemble_vector(_b, L, constants, coeffs);
  VecRestoreArray(b_local, &array);
  VecGhostRestoreLocalForm(b, &b_local);
}

/// Assemble linear form into an already allocated PETSc vector. Ghost
/// contributions are not accumulated (not sent to owner). Caller is
/// responsible for calling VecGhostUpdateBegin/End.
///
/// @param[in,out] b The PETsc vector to assemble the form into. The
/// vector must already be initialised with the correct size. The
/// process-local contribution of the form is assembled into this
/// vector. It is not zeroed before assembly.
/// @param[in] L The linear form to assemble
template <std::floating_point T>
void assemble_vector(Vec b, const Form<PetscScalar, T>& L)
{
  Vec b_local;
  VecGhostGetLocalForm(b, &b_local);
  PetscInt n = 0;
  VecGetSize(b_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b_local, &array);
  std::span<PetscScalar> _b(array, n);
  fem::assemble_vector(_b, L);
  VecRestoreArray(b_local, &array);
  VecGhostRestoreLocalForm(b, &b_local);
}

// FIXME: clarify how x0 is used
// FIXME: if bcs entries are set

// FIXME: need to pass an array of Vec for x0?
// FIXME: clarify zeroing of vector

/// @brief Modify RHS vector to account for Dirichlet boundary
/// conditions.
///
/// Modify b such that:
///
///   b <- b - scale * A_j (g_j - x0_j)
///
/// where j is a block (nest) index. For a non-blocked problem j = 0. The
/// boundary conditions bcs1 are on the trial spaces V_j. The forms in
/// [a] must have the same test space as L (from which b was built), but the
/// trial space may differ. If x0 is not supplied, then it is treated as
/// zero.
///
/// Ghost contributions are not accumulated (not sent to owner). Caller
/// is responsible for calling VecGhostUpdateBegin/End.
template <std::floating_point T>
void apply_lifting(
    Vec b, const std::vector<std::shared_ptr<const Form<PetscScalar, T>>>& a,
    const std::vector<std::span<const PetscScalar>>& constants,
    const std::vector<std::map<std::pair<IntegralType, int>,
                               std::pair<std::span<const PetscScalar>, int>>>&
        coeffs,
    const std::vector<
        std::vector<std::shared_ptr<const DirichletBC<PetscScalar, T>>>>& bcs1,
    const std::vector<Vec>& x0, PetscScalar scale)
{
  Vec b_local;
  VecGhostGetLocalForm(b, &b_local);
  PetscInt n = 0;
  VecGetSize(b_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b_local, &array);
  std::span<PetscScalar> _b(array, n);

  if (x0.empty())
    fem::apply_lifting(_b, a, constants, coeffs, bcs1, {}, scale);
  else
  {
    std::vector<std::span<const PetscScalar>> x0_ref;
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

    std::vector x0_tmp(x0_ref.begin(), x0_ref.end());
    fem::apply_lifting(_b, a, constants, coeffs, bcs1, x0_tmp, scale);

    for (std::size_t i = 0; i < x0_local.size(); ++i)
    {
      VecRestoreArrayRead(x0_local[i], &x0_array[i]);
      VecGhostRestoreLocalForm(x0[i], &x0_local[i]);
    }
  }

  VecRestoreArray(b_local, &array);
  VecGhostRestoreLocalForm(b, &b_local);
}

// FIXME: clarify how x0 is used
// FIXME: if bcs entries are set

// FIXME: need to pass an array of Vec for x0?
// FIXME: clarify zeroing of vector

/// Modify b such that:
///
///   b <- b - scale * A_j (g_j - x0_j)
///
/// where j is a block (nest) index. For a non-blocked problem j = 0. The
/// boundary conditions bcs1 are on the trial spaces V_j. The forms in
/// [a] must have the same test space as L (from which b was built), but the
/// trial space may differ. If x0 is not supplied, then it is treated as
/// zero.
///
/// Ghost contributions are not accumulated (not sent to owner). Caller
/// is responsible for calling VecGhostUpdateBegin/End.
template <std::floating_point T>
void apply_lifting(
    Vec b,
    const std::vector<std::shared_ptr<const Form<PetscScalar, double>>>& a,
    const std::vector<
        std::vector<std::shared_ptr<const DirichletBC<PetscScalar, double>>>>&
        bcs1,
    const std::vector<Vec>& x0, PetscScalar scale)
{
  Vec b_local;
  VecGhostGetLocalForm(b, &b_local);
  PetscInt n = 0;
  VecGetSize(b_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b_local, &array);
  std::span<PetscScalar> _b(array, n);

  if (x0.empty())
    fem::apply_lifting<PetscScalar>(_b, a, bcs1, {}, scale);
  else
  {
    std::vector<std::span<const PetscScalar>> x0_ref;
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

    std::vector x0_tmp(x0_ref.begin(), x0_ref.end());
    fem::apply_lifting<PetscScalar>(_b, a, bcs1, x0_tmp, scale);

    for (std::size_t i = 0; i < x0_local.size(); ++i)
    {
      VecRestoreArrayRead(x0_local[i], &x0_array[i]);
      VecGhostRestoreLocalForm(x0[i], &x0_local[i]);
    }
  }

  VecRestoreArray(b_local, &array);
  VecGhostRestoreLocalForm(b, &b_local);
}

// -- Setting bcs ------------------------------------------------------------

// FIXME: Move these function elsewhere?

// FIXME: clarify x0
// FIXME: clarify what happens with ghosts

/// Set bc values in owned (local) part of the PETSc vector, multiplied
/// by 'scale'. The vectors b and x0 must have the same local size. The
/// bcs should be on (sub-)spaces of the form L that b represents.
template <std::floating_point T>
void set_bc(
    Vec b,
    const std::vector<std::shared_ptr<const DirichletBC<PetscScalar, T>>>& bcs,
    const Vec x0, PetscScalar scale = 1)
{
  PetscInt n = 0;
  VecGetLocalSize(b, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b, &array);
  std::span<PetscScalar> _b(array, n);
  if (x0)
  {
    Vec x0_local;
    VecGhostGetLocalForm(x0, &x0_local);
    PetscInt n = 0;
    VecGetSize(x0_local, &n);
    const PetscScalar* array = nullptr;
    VecGetArrayRead(x0_local, &array);
    std::span<const PetscScalar> _x0(array, n);
    fem::set_bc(_b, bcs, _x0, scale);
    VecRestoreArrayRead(x0_local, &array);
    VecGhostRestoreLocalForm(x0, &x0_local);
  }
  else
    fem::set_bc(_b, bcs, scale);

  VecRestoreArray(b, &array);
}

} // namespace petsc
} // namespace dolfinx::fem
