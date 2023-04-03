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
  pattern.assemble();
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

  std::shared_ptr mesh = V[0][0]->mesh();
  assert(mesh);
  auto topology = mesh->topology();
  assert(topology);
  const int tdim = topology->dim();

  // Build sparsity pattern for each block
  std::vector<std::vector<std::unique_ptr<la::SparsityPattern>>> patterns(
      V[0].size());
  for (std::size_t row = 0; row < V[0].size(); ++row)
  {
    for (std::size_t col = 0; col < V[1].size(); ++col)
    {
      const std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps
          = {{V[0][row]->dofmap()->index_map, V[1][col]->dofmap()->index_map}};
      const std::array bs = {V[0][row]->dofmap()->index_map_bs(),
                             V[1][col]->dofmap()->index_map_bs()};
      if (const Form<PetscScalar, T>* form = a[row][col]; form)
      {
        // Create sparsity pattern for block
        patterns[row].push_back(std::make_unique<la::SparsityPattern>(
            mesh->comm(), index_maps, bs));

        // Build sparsity pattern for block
        assert(V[0][row]->dofmap());
        assert(V[1][col]->dofmap());
        std::array<std::reference_wrapper<const DofMap>, 2> dofmaps{
            *V[0][row]->dofmap(), *V[1][col]->dofmap()};
        assert(patterns[row].back());
        auto& sp = patterns[row].back();
        assert(sp);

        if (form->num_integrals(IntegralType::cell) > 0)
        {
          auto map = topology->index_map(tdim);
          assert(map);
          std::vector<std::int32_t> c(map->size_local(), 0);
          std::iota(c.begin(), c.end(), 0);
          sparsitybuild::cells(*sp, c, dofmaps);
        }

        if (form->num_integrals(IntegralType::interior_facet) > 0)
        {
          // Loop over owned facets
          mesh->topology_mutable()->create_entities(tdim - 1);
          auto f_to_c = topology->connectivity(tdim - 1, tdim);
          if (!f_to_c)
          {
            throw std::runtime_error(
                "Facet-cell connectivity has not been computed.");
          }
          auto map = topology->index_map(tdim - 1);
          assert(map);
          std::vector<std::int32_t> facets;
          facets.reserve(2 * map->size_local());
          for (int f = 0; f < map->size_local(); ++f)
            if (auto cells = f_to_c->links(f); cells.size() == 2)
              facets.insert(facets.end(), {cells[0], cells[1]});
          sparsitybuild::interior_facets(*sp, facets, dofmaps);
        }

        if (form->num_integrals(IntegralType::exterior_facet) > 0)
        {
          // Loop over owned facets
          mesh->topology_mutable()->create_entities(tdim - 1);
          auto connectivity = topology->connectivity(tdim - 1, tdim);
          if (!connectivity)
          {
            throw std::runtime_error(
                "Facet-cell connectivity has not been computed.");
          }
          auto map = topology->index_map(tdim - 1);
          assert(map);
          std::vector<std::int32_t> cells;
          for (int f = 0; f < map->size_local(); ++f)
            if (auto c = connectivity->links(f); c.size() == 1)
              cells.push_back(c[0]);
          sparsitybuild::cells(*sp, cells, dofmaps);
        }
      }
      else
        patterns[row].push_back(nullptr);
    }
  }

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
  pattern.assemble();

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

/// Create nested (MatNest) matrix
///
/// The caller is responsible for destroying the Mat object
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
  const int rows = a.size();
  const int cols = a.front().size();
  std::vector<Mat> mats(rows * cols, nullptr);
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      if (const Form<PetscScalar, T>* form = a[i][j]; form)
        mats[i * cols + j] = create_matrix(*form, _types[i][j]);
    }
  }

  // Initialise block (MatNest) matrix
  Mat A;
  MatCreate(V[0][0]->mesh()->comm(), &A);
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
