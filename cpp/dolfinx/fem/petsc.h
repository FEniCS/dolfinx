// Copyright (C) 2018-2026 Garth N. Wells and Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_PETSC

#include "Form.h"
#include "Function.h"
#include "assembler.h"
#include "utils.h"
#include <cassert>
#include <concepts>
#include <cstdint>
#include <dolfinx/la/petsc.h>
#include <format>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <petscmat.h>
#include <petscvec.h>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::fem
{
template <dolfinx::scalar T, std::floating_point U>
class DirichletBC;

/// @brief Helper functions for assembly into PETSc data structures
namespace petsc
{
/// @brief Create a matrix
/// @param[in] a A bilinear form
/// @param[in] type The PETSc matrix type to create
/// @return A sparse matrix with a layout and sparsity that matches the
/// bilinear form. The caller is responsible for destroying the Mat
/// object.
template <std::floating_point T>
Mat create_matrix(const Form<PetscScalar, T>& a,
                  std::optional<std::string> type = std::nullopt)
{
  la::SparsityPattern pattern = fem::create_sparsity_pattern(a);
  pattern.finalize();
  return la::petsc::create_matrix(a.mesh()->comm(), pattern, type);
}

/// @brief Initialise a monolithic matrix for an array of bilinear
/// forms.
///
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
    std::optional<std::string> type = std::nullopt)
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
    throw std::invalid_argument("Could not find a Mesh.");

  // Compute offsets for the fields
  std::array<std::vector<std::pair<
                 std::reference_wrapper<const common::IndexMap>, int>>,
             2>
      maps;
  for (std::size_t d = 0; d < 2; ++d)
  {
    for (auto& space : V[d])
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

  // TODO: Index map concatenation has already been computed inside
  // the SparsityPattern constructor, but we also need it here to
  // build the PETSc local-to-global map. Compute outside and pass
  // into SparsityPattern constructor.

  // Initialise matrix
  Mat A = la::petsc::create_matrix(mesh->comm(), pattern, type);

  // Create row and column local-to-global maps (field0, field1, field2,
  // etc), i.e. ghosts of field0 appear before owned indices of field1
  std::array<std::vector<PetscInt>, 2> _maps;
  for (int d = 0; d < 2; ++d)
  {
    if (d == 1 and V[0] == V[1])
    {
      // Row and column spaces are identical, so the concatenated
      // index map for d=1 is identical to the one already computed
      // for d=0 -- reuse it rather than paying for a second,
      // communication-heavy call to stack_index_maps.
      _maps[1] = _maps[0];
      continue;
    }

    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& map
        = maps[d];
    std::vector<PetscInt>& _map = _maps[d];

    // Concatenate the block index map in the row and column directions
    const auto [rank_offset, local_offset, ghosts, _]
        = common::stack_index_maps(map);
    const std::size_t num_ghosts
        = std::accumulate(ghosts.begin(), ghosts.end(), std::size_t(0),
                          [](std::size_t n, auto& g) { return n + g.size(); });
    _map.reserve(local_offset.back() + num_ghosts);
    for (std::size_t f = 0; f < map.size(); ++f)
    {
      auto offset = local_offset[f];
      const common::IndexMap& imap = map[f].first.get();
      int bs = map[f].second;
      auto owned
          = std::views::iota(std::int32_t(0), bs * imap.size_local())
            | std::views::transform([offset, rank_offset](std::int32_t i)
                                    { return i + rank_offset + offset; });
      _map.insert(_map.end(), owned.begin(), owned.end());
      _map.insert(_map.end(), ghosts[f].begin(), ghosts[f].end());
    }
  }

  // Create PETSc local-to-global map/index sets and attach to matrix
  ISLocalToGlobalMapping petsc_local_to_global0;
  common::petsc::check(ISLocalToGlobalMappingCreate(
                           MPI_COMM_SELF, 1, _maps[0].size(), _maps[0].data(),
                           PETSC_COPY_VALUES, &petsc_local_to_global0),
                       "ISLocalToGlobalMappingCreate");
  if (V[0] == V[1])
  {
    common::petsc::check(MatSetLocalToGlobalMapping(A, petsc_local_to_global0,
                                                    petsc_local_to_global0),
                         "MatSetLocalToGlobalMapping");
    common::petsc::check(ISLocalToGlobalMappingDestroy(&petsc_local_to_global0),
                         "ISLocalToGlobalMappingDestroy");
  }
  else
  {
    ISLocalToGlobalMapping petsc_local_to_global1;
    common::petsc::check(ISLocalToGlobalMappingCreate(
                             MPI_COMM_SELF, 1, _maps[1].size(), _maps[1].data(),
                             PETSC_COPY_VALUES, &petsc_local_to_global1),
                         "ISLocalToGlobalMappingCreate");
    common::petsc::check(MatSetLocalToGlobalMapping(A, petsc_local_to_global0,
                                                    petsc_local_to_global1),
                         "MatSetLocalToGlobalMapping");
    common::petsc::check(ISLocalToGlobalMappingDestroy(&petsc_local_to_global0),
                         "ISLocalToGlobalMappingDestroy");
    common::petsc::check(ISLocalToGlobalMappingDestroy(&petsc_local_to_global1),
                         "ISLocalToGlobalMappingDestroy");
  }

  return A;
}

/// @brief Create nested (MatNest) matrix.
///
/// @note The caller is responsible for destroying the Mat object.
template <std::floating_point T>
Mat create_matrix_nest(
    const std::vector<std::vector<const Form<PetscScalar, T>*>>& a,
    std::optional<std::vector<std::vector<std::optional<std::string>>>> types)
{
  if (a.empty())
    throw std::invalid_argument(
        "Rectangular array of forms must be non-empty.");

  // Extract and check row/column ranges
  auto V = fem::common_function_spaces(extract_function_spaces(a));

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
        if (types)
          mats[i * cols + j] = create_matrix(*form, types->at(i).at(j));
        else
          mats[i * cols + j] = create_matrix(*form, std::nullopt);
        mesh = form->mesh();
      }
    }
  }

  if (!mesh)
    throw std::invalid_argument("Could not find a Mesh.");

  // Initialise block (MatNest) matrix. On error, destroy the
  // already-created sub-matrices in `mats` before propagating, since
  // the nest (which would otherwise take joint ownership of them) was
  // never successfully assembled.
  Mat A;
  try
  {
    common::petsc::check(MatCreate(mesh->comm(), &A), "MatCreate");
    common::petsc::check(MatSetType(A, MATNEST), "MatSetType");
    common::petsc::check(
        MatNestSetSubMats(A, rows, nullptr, cols, nullptr, mats.data()),
        "MatNestSetSubMats");
    common::petsc::check(MatSetUp(A), "MatSetUp");
  }
  catch (...)
  {
    for (Mat& m : mats)
      if (m)
        common::petsc::check(MatDestroy(&m), "MatDestroy");
    throw;
  }

  // De-reference Mat objects
  for (Mat& m : mats)
    if (m)
      common::petsc::check(MatDestroy(&m), "MatDestroy");

  return A;
}

/// @brief Initialise monolithic vector. Vector is not zeroed.
///
/// The caller is responsible for destroying the Vec object
Vec create_vector_block(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps);

/// @brief Create nested (VecNest) vector. Vector is not zeroed.
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
  common::petsc::check(VecGhostGetLocalForm(b, &b_local),
                       "VecGhostGetLocalForm");
  PetscInt n = 0;
  common::petsc::check(VecGetSize(b_local, &n), "VecGetSize");
  PetscScalar* array = nullptr;
  common::petsc::check(VecGetArray(b_local, &array), "VecGetArray");
  std::span<PetscScalar> _b(array, n);
  fem::assemble_vector(_b, L, constants, coeffs);
  common::petsc::check(VecRestoreArray(b_local, &array), "VecRestoreArray");
  common::petsc::check(VecGhostRestoreLocalForm(b, &b_local),
                       "VecGhostRestoreLocalForm");
}

/// @brief Assemble linear form into an already allocated PETSc vector.
///
/// Ghost contributions are not accumulated (not sent to owner). Caller
/// is responsible for calling `VecGhostUpdateBegin`/`End`.
///
/// @param[in,out] b Vector to assemble the form into. The vector must
/// already be initialised with the correct size. The process-local
/// contribution of the form is assembled into this vector. It is not
/// zeroed before assembly.
/// @param[in] L Linear form to assemble.
template <std::floating_point T>
void assemble_vector(Vec b, const Form<PetscScalar, T>& L)
{
  Vec b_local;
  common::petsc::check(VecGhostGetLocalForm(b, &b_local),
                       "VecGhostGetLocalForm");
  PetscInt n = 0;
  common::petsc::check(VecGetSize(b_local, &n), "VecGetSize");
  PetscScalar* array = nullptr;
  common::petsc::check(VecGetArray(b_local, &array), "VecGetArray");
  std::span<PetscScalar> _b(array, n);
  fem::assemble_vector(_b, L);
  common::petsc::check(VecRestoreArray(b_local, &array), "VecRestoreArray");
  common::petsc::check(VecGhostRestoreLocalForm(b, &b_local),
                       "VecGhostRestoreLocalForm");
}

// FIXME: clarify zeroing of vector

/// @brief Modify RHS vector to account for Dirichlet boundary
/// conditions.
///
/// Modify b such that:
///
///   b <- b - alpha * A_j (g_j - x0_j)
///
/// where j is a block (nest) index. For a non-blocked problem j = 0. The
/// boundary conditions bcs1 are on the trial spaces V_j. The forms in
/// [a] must have the same test space as L (from which b was built), but the
/// trial space may differ. If x0 is not supplied, then it is treated as
/// zero.
///
/// Ghost contributions are not accumulated (not sent to owner). Caller
/// is responsible for calling VecGhostUpdateBegin/End.
///
/// @param[in,out] b Vector to modify by lifting.
/// @param[in] a Bilinear forms, one per block `j`. A `std::nullopt`
/// entry skips that block.
/// @param[in] constants Constants that appear in each form in `a`, one
/// entry per block `j`.
/// @param[in] coeffs Coefficients that appear in each form in `a`, one
/// entry per block `j`.
/// @param[in] bcs1 Boundary conditions on the trial space `V_j` for
/// each block `j`.
/// @param[in] x0 Vectors used in the lifting, one per block `j`. If
/// empty, `x0_j` is treated as zero for every block. Otherwise must
/// have the same length as `a`.
/// @param[in] alpha Scaling to apply.
template <std::floating_point T>
void apply_lifting(
    Vec b,
    std::vector<
        std::optional<std::reference_wrapper<const Form<PetscScalar, T>>>>
        a,
    const std::vector<std::span<const PetscScalar>>& constants,
    const std::vector<std::map<std::pair<IntegralType, int>,
                               std::pair<std::span<const PetscScalar>, int>>>&
        coeffs,
    const std::vector<
        std::vector<std::reference_wrapper<const DirichletBC<PetscScalar, T>>>>&
        bcs1,
    const std::vector<Vec>& x0, PetscScalar alpha)
{
  if (!x0.empty() and x0.size() != a.size())
    throw std::invalid_argument("Mismatch between x0 and a in apply_lifting.");

  Vec b_local;
  common::petsc::check(VecGhostGetLocalForm(b, &b_local),
                       "VecGhostGetLocalForm");
  PetscInt n = 0;
  common::petsc::check(VecGetSize(b_local, &n), "VecGetSize");
  PetscScalar* array = nullptr;
  common::petsc::check(VecGetArray(b_local, &array), "VecGetArray");
  std::span<PetscScalar> _b(array, n);

  if (x0.empty())
    fem::apply_lifting(_b, a, constants, coeffs, bcs1, {}, alpha);
  else
  {
    std::vector<std::span<const PetscScalar>> x0_ref;
    std::vector<Vec> x0_local(a.size());
    std::vector<const PetscScalar*> x0_array(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
      assert(x0[i]);
      common::petsc::check(VecGhostGetLocalForm(x0[i], &x0_local[i]),
                           "VecGhostGetLocalForm");
      PetscInt n0 = 0;
      common::petsc::check(VecGetSize(x0_local[i], &n0), "VecGetSize");
      common::petsc::check(VecGetArrayRead(x0_local[i], &x0_array[i]),
                           "VecGetArrayRead");
      x0_ref.emplace_back(x0_array[i], n0);
    }

    fem::apply_lifting(_b, a, constants, coeffs, bcs1, x0_ref, alpha);

    for (std::size_t i = 0; i < x0_local.size(); ++i)
    {
      common::petsc::check(VecRestoreArrayRead(x0_local[i], &x0_array[i]),
                           "VecRestoreArrayRead");
      common::petsc::check(VecGhostRestoreLocalForm(x0[i], &x0_local[i]),
                           "VecGhostRestoreLocalForm");
    }
  }

  common::petsc::check(VecRestoreArray(b_local, &array), "VecRestoreArray");
  common::petsc::check(VecGhostRestoreLocalForm(b, &b_local),
                       "VecGhostRestoreLocalForm");
}

// FIXME: clarify zeroing of vector

/// @brief Modify RHS vector to account for Dirichlet boundary
/// conditions.
///
/// Modify b such that:
///
///   b <- b - alpha * A_j (g_j - x0_j)
///
/// where j is a block (nest) index. For a non-blocked problem j = 0. The
/// boundary conditions bcs1 are on the trial spaces V_j. The forms in
/// [a] must have the same test space as L (from which b was built), but the
/// trial space may differ. If x0 is not supplied, then it is treated as
/// zero.
///
/// Ghost contributions are not accumulated (not sent to owner). Caller
/// is responsible for calling VecGhostUpdateBegin/End.
///
/// @param[in,out] b Vector to modify by lifting.
/// @param[in] a Bilinear forms, one per block `j`. A `std::nullopt`
/// entry skips that block.
/// @param[in] bcs1 Boundary conditions on the trial space `V_j` for
/// each block `j`.
/// @param[in] x0 Vectors used in the lifting, one per block `j`. If
/// empty, `x0_j` is treated as zero for every block. Otherwise must
/// have the same length as `a`.
/// @param[in] alpha Scaling to apply.
template <std::floating_point T>
void apply_lifting(
    Vec b,
    const std::vector<
        std::optional<std::reference_wrapper<const Form<PetscScalar, T>>>>& a,
    const std::vector<
        std::vector<std::reference_wrapper<const DirichletBC<PetscScalar, T>>>>&
        bcs1,
    const std::vector<Vec>& x0, PetscScalar alpha)
{
  if (!x0.empty() and x0.size() != a.size())
    throw std::invalid_argument("Mismatch between x0 and a in apply_lifting.");

  Vec b_local;
  common::petsc::check(VecGhostGetLocalForm(b, &b_local),
                       "VecGhostGetLocalForm");
  PetscInt n = 0;
  common::petsc::check(VecGetSize(b_local, &n), "VecGetSize");
  PetscScalar* array = nullptr;
  common::petsc::check(VecGetArray(b_local, &array), "VecGetArray");
  std::span<PetscScalar> _b(array, n);

  if (x0.empty())
    fem::apply_lifting(_b, a, bcs1, {}, alpha);
  else
  {
    std::vector<std::span<const PetscScalar>> x0_ref;
    std::vector<Vec> x0_local(a.size());
    std::vector<const PetscScalar*> x0_array(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
      assert(x0[i]);
      common::petsc::check(VecGhostGetLocalForm(x0[i], &x0_local[i]),
                           "VecGhostGetLocalForm");
      PetscInt n0 = 0;
      common::petsc::check(VecGetSize(x0_local[i], &n0), "VecGetSize");
      common::petsc::check(VecGetArrayRead(x0_local[i], &x0_array[i]),
                           "VecGetArrayRead");
      x0_ref.emplace_back(x0_array[i], n0);
    }

    fem::apply_lifting(_b, a, bcs1, x0_ref, alpha);

    for (std::size_t i = 0; i < x0_local.size(); ++i)
    {
      common::petsc::check(VecRestoreArrayRead(x0_local[i], &x0_array[i]),
                           "VecRestoreArrayRead");
      common::petsc::check(VecGhostRestoreLocalForm(x0[i], &x0_local[i]),
                           "VecGhostRestoreLocalForm");
    }
  }

  common::petsc::check(VecRestoreArray(b_local, &array), "VecRestoreArray");
  common::petsc::check(VecGhostRestoreLocalForm(b, &b_local),
                       "VecGhostRestoreLocalForm");
}

// -- Setting bcs ------------------------------------------------------------

// FIXME: Move these function elsewhere?

/// @brief Entries in `b` that are constrained by a Dirichlet boundary
/// conditions are set to `alpha * (x_bc - x0)`, where `x_bc` is the
/// (interpolated) boundary condition value.
///
/// @param[in] b The vector to apply the boundary condition to. The local
/// (owned) part of this vector is modified. The user is responsible for
/// scattering the changes to the ghost part of the vector if necessary.
/// @param[in] bcs The boundary conditions to apply.
/// @param[in] x0 Optional vector used in computing the value to set. If
/// not provided it is treated as zero. The local (owned) part of this vector is
/// used.
/// @param[in] alpha Scaling to apply.
template <std::floating_point T>
void set_bc(Vec b,
            const std::vector<
                std::reference_wrapper<const DirichletBC<PetscScalar, T>>>& bcs,
            std::optional<const Vec> x0, PetscScalar alpha = 1)
{
  PetscInt n = 0;
  common::petsc::check(VecGetLocalSize(b, &n), "VecGetLocalSize");
  PetscScalar* array = nullptr;
  common::petsc::check(VecGetArray(b, &array), "VecGetArray");
  std::span<PetscScalar> _b(array, n);
  if (x0.has_value())
  {
    Vec x0_local;
    common::petsc::check(VecGhostGetLocalForm(x0.value(), &x0_local),
                         "VecGhostGetLocalForm");
    PetscInt n0 = 0;
    common::petsc::check(VecGetSize(x0_local, &n0), "VecGetSize");
    const PetscScalar* x0_array = nullptr;
    common::petsc::check(VecGetArrayRead(x0_local, &x0_array),
                         "VecGetArrayRead");
    std::span<const PetscScalar> _x0(x0_array, n0);
    for (auto& bc : bcs)
      bc.get().set(_b, _x0, alpha);
    common::petsc::check(VecRestoreArrayRead(x0_local, &x0_array),
                         "VecRestoreArrayRead");
    common::petsc::check(VecGhostRestoreLocalForm(x0.value(), &x0_local),
                         "VecGhostRestoreLocalForm");
  }
  else
  {
    for (auto& bc : bcs)
      bc.get().set(_b, std::nullopt, alpha);
  }
  common::petsc::check(VecRestoreArray(b, &array), "VecRestoreArray");
}

// -- Nonlinear problem assembly ---------------------------------------------

namespace impl
{
/// @brief Copy a vector into the degrees-of-freedom of a function.
/// @param[in] x Vector to copy from. Must be ghosted, with up-to-date
/// ghost values.
/// @param[out] u Function to copy into.
template <std::floating_point T>
void assign(const Vec x, Function<PetscScalar, T>& u)
{
  Vec x_local = nullptr;
  common::petsc::check(VecGhostGetLocalForm(x, &x_local),
                       "VecGhostGetLocalForm");
  PetscInt n = 0;
  common::petsc::check(VecGetSize(x_local, &n), "VecGetSize");

  std::span<PetscScalar> _u = u.x()->array();
  if (static_cast<std::size_t>(n) != _u.size())
  {
    throw std::runtime_error(std::format(
        "Vector has {} local entries, function has {}.", n, _u.size()));
  }

  const PetscScalar* array = nullptr;
  common::petsc::check(VecGetArrayRead(x_local, &array), "VecGetArrayRead");
  std::ranges::copy(std::span<const PetscScalar>(array, n), _u.begin());
  common::petsc::check(VecRestoreArrayRead(x_local, &array),
                       "VecRestoreArrayRead");
  common::petsc::check(VecGhostRestoreLocalForm(x, &x_local),
                       "VecGhostRestoreLocalForm");
}

/// @brief Zero `A`, assemble `a` into it with `bcs` applied, set the
/// unit diagonal on rows constrained by `bcs`, and finalise assembly.
/// @param[out] A Matrix to assemble into.
/// @param[in] a Bilinear form to assemble.
/// @param[in] bcs Dirichlet boundary conditions.
template <std::floating_point T>
void assemble_operator(
    Mat A, const Form<PetscScalar, T>& a,
    const std::vector<
        std::reference_wrapper<const DirichletBC<PetscScalar, T>>>& bcs)
{
  common::petsc::check(MatZeroEntries(A), "MatZeroEntries");
  fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A, ADD_VALUES), a, bcs);

  // The unit diagonal is only meaningful when the rows and columns are
  // indexed by the same space
  if (a.function_spaces()[0] == a.function_spaces()[1])
  {
    // Flush to switch from adding to inserting
    common::petsc::check(MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY),
                         "MatAssemblyBegin");
    common::petsc::check(MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY),
                         "MatAssemblyEnd");
    fem::set_diagonal(la::petsc::Matrix::set_fn(A, INSERT_VALUES),
                      *a.function_spaces()[0], bcs);
  }

  common::petsc::check(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY),
                       "MatAssemblyBegin");
  common::petsc::check(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
}
} // namespace impl

/// @brief Assemble the residual \f$F(x)\f$ of a nonlinear problem into
/// `b`, with Dirichlet conditions applied.
///
/// Intended as the body of the residual callback of
/// nls::petsc::SNESSolver, which passes the point to evaluate at `x`
/// and the vector to assemble into `b`:
/// @code
/// solver.set_F([&](const Vec x, Vec b)
///              { assemble_residual(x, b, F, J, bcs, u); }, b_layout);
/// @endcode
///
/// Entries of `b` constrained by `bcs` are set to `x - g`, so that a
/// Newton update drives `x` to the boundary condition value `g`.
///
/// @param[in] x Point at which to evaluate the residual, e.g. a line
/// search trial point. Must be ghosted. Its ghost values are updated
/// before use.
/// @param[out] b Vector to assemble into, which is the one the solver
/// passed to the callback and not necessarily the one registered with
/// set_F. Zeroed first, and its ghost values are updated on return.
/// @param[in] F Residual form.
/// @param[in] J Jacobian form, used to lift `bcs`.
/// @param[in] bcs Dirichlet boundary conditions.
/// @param[out] u Function that `F` and `J` hold as a coefficient. Its
/// degrees-of-freedom are set to `x` before assembly.
template <std::floating_point T>
void assemble_residual(
    const Vec x, Vec b, const Form<PetscScalar, T>& F,
    const Form<PetscScalar, T>& J,
    const std::vector<
        std::reference_wrapper<const DirichletBC<PetscScalar, T>>>& bcs,
    Function<PetscScalar, T>& u)
{
  common::petsc::check(VecGhostUpdateBegin(x, INSERT_VALUES, SCATTER_FORWARD),
                       "VecGhostUpdateBegin");
  common::petsc::check(VecGhostUpdateEnd(x, INSERT_VALUES, SCATTER_FORWARD),
                       "VecGhostUpdateEnd");
  impl::assign(x, u);

  // Zero the local form, as assembly accumulates into ghost entries
  Vec b_local = nullptr;
  common::petsc::check(VecGhostGetLocalForm(b, &b_local),
                       "VecGhostGetLocalForm");
  common::petsc::check(VecZeroEntries(b_local), "VecZeroEntries");
  common::petsc::check(VecGhostRestoreLocalForm(b, &b_local),
                       "VecGhostRestoreLocalForm");

  assemble_vector(b, F);

  std::vector<std::optional<std::reference_wrapper<const Form<PetscScalar, T>>>>
      a{J};
  std::vector<
      std::vector<std::reference_wrapper<const DirichletBC<PetscScalar, T>>>>
      bcs1{bcs};
  apply_lifting(b, a, bcs1, std::vector<Vec>{x}, -1);

  common::petsc::check(VecGhostUpdateBegin(b, ADD_VALUES, SCATTER_REVERSE),
                       "VecGhostUpdateBegin");
  common::petsc::check(VecGhostUpdateEnd(b, ADD_VALUES, SCATTER_REVERSE),
                       "VecGhostUpdateEnd");

  set_bc(b, bcs, x, -1);

  common::petsc::check(VecGhostUpdateBegin(b, INSERT_VALUES, SCATTER_FORWARD),
                       "VecGhostUpdateBegin");
  common::petsc::check(VecGhostUpdateEnd(b, INSERT_VALUES, SCATTER_FORWARD),
                       "VecGhostUpdateEnd");
}

/// @brief Assemble the Jacobian \f$dF/dx\f$ of a nonlinear problem into
/// `Jmat`, and a preconditioner into `Pmat`.
///
/// Intended as the body of the Jacobian callback of
/// nls::petsc::SNESSolver, which passes the point to evaluate at `x`
/// and the matrices to assemble into `Jmat` and `Pmat`:
/// @code
/// solver.set_J([&](const Vec x, Mat Jmat, Mat Pmat)
///              { assemble_jacobian(x, Jmat, Pmat, J, bcs, u); },
///              A_layout);
/// @endcode
///
/// Rows and columns constrained by `bcs` are zeroed, and for a form
/// whose test and trial spaces are the same a unit diagonal is set on
/// the constrained rows, matching the residual assembled by
/// assemble_residual.
///
/// @param[in] x Point at which to evaluate the Jacobian, e.g. a line
/// search trial point. Must be ghosted. Its ghost values are updated
/// before use.
/// @param[out] Jmat Matrix to assemble the Jacobian into, which is the
/// one the solver passed to the callback and not necessarily the one
/// registered with set_J. Zeroed first.
/// @param[out] Pmat Matrix to assemble the preconditioner into. Zeroed
/// first. Unused, and may be `nullptr`, if `P` is not given.
/// @param[in] J Jacobian form.
/// @param[in] bcs Dirichlet boundary conditions.
/// @param[out] u Function that `J` and `P` hold as a coefficient. Its
/// degrees-of-freedom are set to `x` before assembly.
/// @param[in] P Preconditioner form. If not given, `Pmat` is left
/// alone and PETSc preconditions with the Jacobian.
template <std::floating_point T>
void assemble_jacobian(
    const Vec x, Mat Jmat, Mat Pmat, const Form<PetscScalar, T>& J,
    const std::vector<
        std::reference_wrapper<const DirichletBC<PetscScalar, T>>>& bcs,
    Function<PetscScalar, T>& u, const Form<PetscScalar, T>* P = nullptr)
{
  common::petsc::check(VecGhostUpdateBegin(x, INSERT_VALUES, SCATTER_FORWARD),
                       "VecGhostUpdateBegin");
  common::petsc::check(VecGhostUpdateEnd(x, INSERT_VALUES, SCATTER_FORWARD),
                       "VecGhostUpdateEnd");
  impl::assign(x, u);

  impl::assemble_operator(Jmat, J, bcs);
  if (P)
    impl::assemble_operator(Pmat, *P, bcs);
}

} // namespace petsc
} // namespace dolfinx::fem

#endif
