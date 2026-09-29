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
#include "pack.h"
#include "sparsitypattern.h"
#include <cassert>
#include <concepts>
#include <cstdint>
#include <dolfinx/la/petsc.h>
#include <format>
#include <functional>
#include <iterator>
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

  // TODO: Index map concatenation has already been computed inside
  // the SparsityPattern constructor, but we also need it here to
  // build the PETSc local-to-global map. Compute outside and pass
  // into SparsityPattern constructor.

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

  // Create the local-to-global maps on the mesh communicator. MATIS
  // requires them to share the matrix communicator
  ISLocalToGlobalMapping l2g0 = nullptr, l2g1 = nullptr;
  common::petsc::check(
      ISLocalToGlobalMappingCreate(mesh->comm(), 1, _maps[0].size(),
                                   _maps[0].data(), PETSC_COPY_VALUES, &l2g0),
      "ISLocalToGlobalMappingCreate");
  if (V[0] != V[1])
  {
    common::petsc::check(
        ISLocalToGlobalMappingCreate(mesh->comm(), 1, _maps[1].size(),
                                     _maps[1].data(), PETSC_COPY_VALUES, &l2g1),
        "ISLocalToGlobalMappingCreate");
  }

  // Initialise the matrix. MATIS builds its preallocation from the
  // maps, so they are passed to the constructor
  Mat A = la::petsc::create_matrix(mesh->comm(), pattern, type, l2g0,
                                   l2g1 ? l2g1 : l2g0);
  common::petsc::check(ISLocalToGlobalMappingDestroy(&l2g0),
                       "ISLocalToGlobalMappingDestroy");
  if (l2g1)
  {
    common::petsc::check(ISLocalToGlobalMappingDestroy(&l2g1),
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
/// conditions, with the constrained dofs and their values given as
/// arrays.
///
/// Modify b such that:
///
///   b <- b - alpha * A_j (g_j - x0_j)
///
/// where j is a block (nest) index. For a non-blocked problem j = 0.
/// The forms in [a] must have the same test space as L (from which b
/// was built), but the trial space may differ. If x0 is not supplied,
/// then it is treated as zero.
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
/// @param[in] bc_markers1 Constrained dof markers on the trial space
/// `V_j` of each block `j` (owned and ghost, unrolled). An empty entry
/// means block `j` has no constraints.
/// @param[in] bc_values1 Boundary condition values `g_j` on `V_j`,
/// read where `bc_markers1[j]` is non-zero. Same length as
/// `bc_markers1[j]`.
/// @param[in] x0 Vectors used in the lifting, one per block `j`. If
/// empty, `x0_j` is treated as zero for every block. Otherwise must
/// have the same length as `a`.
/// @param[in] alpha Scaling to apply.
template <std::floating_point T>
void apply_lifting(
    Vec b,
    const std::vector<
        std::optional<std::reference_wrapper<const Form<PetscScalar, T>>>>& a,
    const std::vector<std::span<const PetscScalar>>& constants,
    const std::vector<std::map<std::pair<IntegralType, int>,
                               std::pair<std::span<const PetscScalar>, int>>>&
        coeffs,
    const std::vector<std::span<const std::int8_t>>& bc_markers1,
    const std::vector<std::span<const PetscScalar>>& bc_values1,
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
  {
    fem::apply_lifting(_b, a, constants, coeffs, bc_markers1, bc_values1, {},
                       alpha);
  }
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

    fem::apply_lifting(_b, a, constants, coeffs, bc_markers1, bc_values1,
                       x0_ref, alpha);

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
/// @note Convenience overload for callers that have boundary
/// conditions. It rebuilds the constrained dof markers and values on
/// every call, and should not be called internally by the library;
/// call the overload taking `bc_markers1` and `bc_values1` instead.
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
  std::vector<std::map<std::pair<IntegralType, int>,
                       std::pair<std::vector<PetscScalar>, int>>>
      coeffs;
  std::vector<std::vector<PetscScalar>> constants;
  for (const auto& _a : a)
  {
    if (_a)
    {
      auto coefficients = allocate_coefficient_storage(_a->get());
      pack_coefficients(_a->get(), coefficients);
      coeffs.push_back(coefficients);
      constants.push_back(pack_constants(_a->get()));
    }
    else
    {
      coeffs.emplace_back();
      constants.emplace_back();
    }
  }

  std::vector<std::span<const PetscScalar>> _constants(constants.begin(),
                                                       constants.end());
  std::vector<std::map<std::pair<IntegralType, int>,
                       std::pair<std::span<const PetscScalar>, int>>>
      _coeffs;
  std::ranges::transform(coeffs, std::back_inserter(_coeffs),
                         [](auto& c) { return make_coefficients_span(c); });

  auto [bc_markers1, bc_values1] = fem::impl::bc_lifting_data(a, bcs1);
  apply_lifting(b, a, _constants, _coeffs,
                std::vector<std::span<const std::int8_t>>(bc_markers1.begin(),
                                                          bc_markers1.end()),
                std::vector<std::span<const PetscScalar>>(bc_values1.begin(),
                                                          bc_values1.end()),
                x0, alpha);
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

/// @brief Zero `A`, assemble `a` into it with the rows and columns of
/// constrained dofs zeroed, set the unit diagonal on constrained rows,
/// and finalise assembly.
///
/// Steps:
/// 1. Zero all entries of `A`.
/// 2. Assemble `a`, zeroing the rows marked in `dof_marker0` and the
///    columns marked in `dof_marker1`.
/// 3. If the test and trial spaces are the same object, insert 1 on
///    the diagonal of each locally owned row marked in `dof_marker0`.
///    Ghost rows are set by their owning process.
/// 4. Finalise assembly (`MAT_FINAL_ASSEMBLY`).
///
/// Used for the Jacobian and preconditioner operators in
/// assemble_jacobian().
///
/// @pre `A` has the sparsity and local-to-global maps of `a`, e.g.
/// created by create_matrix().
/// @note Collective on the communicator of `A`.
/// @param[in,out] A Matrix to assemble into. Its previous entries are
/// discarded.
/// @param[in] a Bilinear form to assemble.
/// @param[in] dof_marker0 Constrained dof markers on the test space of
/// `a` (owned and ghost, unrolled), or empty if none are constrained.
/// @param[in] dof_marker1 Constrained dof markers on the trial space of
/// `a` (owned and ghost, unrolled), or empty if none are constrained.
template <std::floating_point T>
void assemble_operator(Mat A, const Form<PetscScalar, T>& a,
                       std::span<const std::int8_t> dof_marker0,
                       std::span<const std::int8_t> dof_marker1)
{
  common::petsc::check(MatZeroEntries(A), "MatZeroEntries");

  // Block and dof indices coincide at block size 1. Avoid the generic
  // blocked PETSc path, which expands these indices before insertion.
  if (a.function_spaces()[0]->dofmap()->index_map_bs() == 1
      and a.function_spaces()[1]->dofmap()->index_map_bs() == 1)
  {
    fem::assemble_matrix(la::petsc::Matrix::set_fn(A, ADD_VALUES), a,
                         dof_marker0, dof_marker1);
  }
  else
  {
    fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A, ADD_VALUES), a,
                         dof_marker0, dof_marker1);
  }

  // The unit diagonal is only meaningful when the rows and columns are
  // indexed by the same space
  if (a.function_spaces()[0] == a.function_spaces()[1])
  {
    // Locally owned constrained rows
    std::shared_ptr<const DofMap> dofmap0
        = a.function_spaces()[0]->dofmaps().front();
    // dof_marker0 is empty when no rows are constrained
    const std::int32_t num_owned
        = dof_marker0.empty()
              ? 0
              : dofmap0->index_map_bs() * dofmap0->index_map->size_local();
    std::vector<std::int32_t> rows;
    std::ranges::copy_if(
        std::views::iota(std::int32_t(0), num_owned), std::back_inserter(rows),
        [&dof_marker0](std::int32_t i) { return dof_marker0[i] != 0; });

    // Assembly zeroed these rows, so adding sets the diagonal. Adding
    // avoids a flush to switch from ADD_VALUES to INSERT_VALUES.
    fem::set_diagonal<PetscScalar>(la::petsc::Matrix::set_fn(A, ADD_VALUES),
                                   rows);
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

  // Constrained dof markers and values g on the trial space of J (owned
  // and ghost), both empty if there are no bcs
  const std::vector<
      std::optional<std::reference_wrapper<const Form<PetscScalar, T>>>>
      a{J};
  auto [bc_markers1, bc_values1] = fem::impl::bc_lifting_data(
      a, std::vector<std::vector<
             std::reference_wrapper<const DirichletBC<PetscScalar, T>>>>{bcs});

  // Lifting: b <- b + J (g - x)
  const std::vector<PetscScalar> constants = pack_constants(J);
  auto coeffs = allocate_coefficient_storage(J);
  pack_coefficients(J, coeffs);
  apply_lifting(b, a, {std::span<const PetscScalar>(constants)},
                {make_coefficients_span(coeffs)},
                {std::span<const std::int8_t>(bc_markers1.front())},
                {std::span<const PetscScalar>(bc_values1.front())}, {x},
                PetscScalar(-1));

  common::petsc::check(VecGhostUpdateBegin(b, ADD_VALUES, SCATTER_REVERSE),
                       "VecGhostUpdateBegin");
  common::petsc::check(VecGhostUpdateEnd(b, ADD_VALUES, SCATTER_REVERSE),
                       "VecGhostUpdateEnd");

  // Set b = x - g on owned constrained dofs
  if (!bcs.empty())
  {
    PetscScalar* b_array = nullptr;
    common::petsc::check(VecGetArray(b, &b_array), "VecGetArray");
    const PetscScalar* x_array = nullptr;
    common::petsc::check(VecGetArrayRead(x, &x_array), "VecGetArrayRead");
    for (auto& bc : bcs)
    {
      auto [dofs, owned] = bc.get().dof_indices();
      for (std::int32_t dof : dofs.first(owned))
        b_array[dof] = x_array[dof] - bc_values1.front()[dof];
    }
    common::petsc::check(VecRestoreArrayRead(x, &x_array),
                         "VecRestoreArrayRead");
    common::petsc::check(VecRestoreArray(b, &b_array), "VecRestoreArray");
  }

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
/// @pre `P`, if given, must have the same function spaces as `J`, so
/// that the constrained dof markers built for `J` apply to it too.
template <std::floating_point T>
void assemble_jacobian(
    const Vec x, Mat Jmat, Mat Pmat, const Form<PetscScalar, T>& J,
    const std::vector<
        std::reference_wrapper<const DirichletBC<PetscScalar, T>>>& bcs,
    Function<PetscScalar, T>& u, const Form<PetscScalar, T>* P = nullptr)
{
  // Checked before any collective call so that all ranks throw together
  if (P and P->function_spaces() != J.function_spaces())
  {
    throw std::invalid_argument(
        "Preconditioner form must have the same function spaces as the "
        "Jacobian form.");
  }

  common::petsc::check(VecGhostUpdateBegin(x, INSERT_VALUES, SCATTER_FORWARD),
                       "VecGhostUpdateBegin");
  common::petsc::check(VecGhostUpdateEnd(x, INSERT_VALUES, SCATTER_FORWARD),
                       "VecGhostUpdateEnd");
  impl::assign(x, u);

  // The markers depend only on the space, so they are built once here
  // and shared by the Jacobian and the preconditioner, which have the
  // same spaces, and by the rows and columns of a square form
  const std::vector<std::int8_t> marker0
      = fem::impl::bc_dof_markers(*J.function_spaces()[0], bcs);
  const bool square = J.function_spaces()[0] == J.function_spaces()[1];
  const std::vector<std::int8_t> marker1
      = square ? std::vector<std::int8_t>()
               : fem::impl::bc_dof_markers(*J.function_spaces()[1], bcs);
  std::span<const std::int8_t> dof_marker0(marker0);
  std::span<const std::int8_t> dof_marker1
      = square ? dof_marker0 : std::span<const std::int8_t>(marker1);

  impl::assemble_operator(Jmat, J, dof_marker0, dof_marker1);
  if (P)
    impl::assemble_operator(Pmat, *P, dof_marker0, dof_marker1);
}

} // namespace petsc
} // namespace dolfinx::fem

#endif
