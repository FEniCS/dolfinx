// Copyright (C) 2018-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "petsc.h"
#include "FunctionSpace.h"
#include "assembler.h"
#include "sparsitybuild.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/petsc.h>
#include <functional>
#include <petscistypes.h>
#include <span>

using namespace dolfinx;

//-----------------------------------------------------------------------------
Mat fem::petsc::create_matrix(const Form<PetscScalar>& a,
                              const std::string& type)
{
  // Build sparsitypattern
  la::SparsityPattern pattern = fem::create_sparsity_pattern(a);

  // Finalise communication
  pattern.assemble();

  return la::petsc::create_matrix(a.mesh()->comm(), pattern, type);
}
//-----------------------------------------------------------------------------
Mat fem::petsc::create_matrix_block(
    const std::vector<std::vector<const Form<PetscScalar>*>>& a,
    const std::string& type)
{
  // Extract and check row/column ranges
  std::array<std::vector<std::shared_ptr<const FunctionSpace>>, 2> V
      = fem::common_function_spaces(extract_function_spaces(a));
  std::array<std::vector<int>, 2> bs_dofs;
  for (std::size_t i = 0; i < 2; ++i)
  {
    for (auto& _V : V[i])
      bs_dofs[i].push_back(_V->dofmap()->bs());
  }

  std::shared_ptr mesh = V[0][0]->mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();

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
      if (const Form<PetscScalar>* form = a[row][col]; form)
      {
        // Create sparsity pattern for block
        patterns[row].push_back(std::make_unique<la::SparsityPattern>(
            mesh->comm(), index_maps, bs));

        // Build sparsity pattern for block
        assert(V[0][row]->dofmap());
        assert(V[1][col]->dofmap());
        std::array<const std::reference_wrapper<const DofMap>, 2> dofmaps{
            *V[0][row]->dofmap(), *V[1][col]->dofmap()};
        assert(patterns[row].back());
        auto& sp = patterns[row].back();
        assert(sp);
        if (form->num_integrals(IntegralType::cell) > 0)
          sparsitybuild::cells(*sp, mesh->topology(), dofmaps);
        if (form->num_integrals(IntegralType::interior_facet) > 0)
        {
          mesh->topology_mutable().create_entities(tdim - 1);
          sparsitybuild::interior_facets(*sp, mesh->topology(), dofmaps);
        }
        if (form->num_integrals(IntegralType::exterior_facet) > 0)
        {
          mesh->topology_mutable().create_entities(tdim - 1);
          sparsitybuild::exterior_facets(*sp, mesh->topology(), dofmaps);
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
//-----------------------------------------------------------------------------
Mat fem::petsc::create_matrix_nest(
    const std::vector<std::vector<const Form<PetscScalar>*>>& a,
    const std::vector<std::vector<std::string>>& types)
{
  // Extract and check row/column ranges
  auto V = fem::common_function_spaces(extract_function_spaces(a));

  std::vector<std::vector<std::string>> _types(
      a.size(), std::vector<std::string>(a[0].size()));
  if (!types.empty())
    _types = types;

  // Loop over each form and create matrix
  const int rows = a.size();
  const int cols = a[0].size();
  std::vector<Mat> mats(rows * cols, nullptr);
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      if (const Form<PetscScalar>* form = a[i][j]; form)
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
//-----------------------------------------------------------------------------
Vec fem::petsc::create_vector_block(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps)
{
  // FIXME: handle constant block size > 1

  auto [rank_offset, local_offset, ghosts_new, ghost_new_owners]
      = common::stack_index_maps(maps);
  std::int32_t local_size = local_offset.back();

  std::vector<std::int64_t> ghosts;
  for (auto& sub_ghost : ghosts_new)
    ghosts.insert(ghosts.end(), sub_ghost.begin(), sub_ghost.end());

  std::vector<int> ghost_owners;
  for (auto& sub_owner : ghost_new_owners)
    ghost_owners.insert(ghost_owners.end(), sub_owner.begin(), sub_owner.end());

  // Create map for combined problem, and create vector
  common::IndexMap index_map(maps[0].first.get().comm(), local_size, ghosts,
                             ghost_owners);

  return la::petsc::create_vector(index_map, 1);
}
//-----------------------------------------------------------------------------
Vec fem::petsc::create_vector_nest(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps)
{
  assert(!maps.empty());

  // Loop over each form and create vector
  std::vector<std::shared_ptr<la::petsc::Vector>> vecs;
  std::vector<Vec> petsc_vecs;
  for (auto& map : maps)
  {
    vecs.push_back(std::make_shared<la::petsc::Vector>(map.first, map.second));
    petsc_vecs.push_back(vecs.back()->vec());
  }

  // Create nested (VecNest) vector
  Vec y;
  VecCreateNest(vecs[0]->comm(), petsc_vecs.size(), nullptr, petsc_vecs.data(),
                &y);
  return y;
}
//-----------------------------------------------------------------------------
void fem::petsc::assemble_vector(
    Vec b, const Form<PetscScalar>& L, std::span<const PetscScalar> constants,
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
  fem::assemble_vector<PetscScalar>(_b, L, constants, coeffs);
  VecRestoreArray(b_local, &array);
  VecGhostRestoreLocalForm(b, &b_local);
}
//-----------------------------------------------------------------------------
void fem::petsc::assemble_vector(Vec b, const Form<PetscScalar>& L)
{
  Vec b_local;
  VecGhostGetLocalForm(b, &b_local);
  PetscInt n = 0;
  VecGetSize(b_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b_local, &array);
  std::span<PetscScalar> _b(array, n);
  fem::assemble_vector<PetscScalar>(_b, L);
  VecRestoreArray(b_local, &array);
  VecGhostRestoreLocalForm(b, &b_local);
}
//-----------------------------------------------------------------------------
void fem::petsc::apply_lifting(
    Vec b, const std::vector<std::shared_ptr<const Form<PetscScalar>>>& a,
    const std::vector<std::span<const PetscScalar>>& constants,
    const std::vector<std::map<std::pair<IntegralType, int>,
                               std::pair<std::span<const PetscScalar>, int>>>&
        coeffs,
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
  std::span<PetscScalar> _b(array, n);

  if (x0.empty())
    fem::apply_lifting<PetscScalar>(_b, a, constants, coeffs, bcs1, {}, scale);
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
    fem::apply_lifting<PetscScalar>(_b, a, constants, coeffs, bcs1, x0_tmp,
                                    scale);

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
void fem::petsc::apply_lifting(
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
//-----------------------------------------------------------------------------
void fem::petsc::set_bc(
    Vec b,
    const std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>& bcs,
    const Vec x0, double scale)
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
    fem::set_bc<PetscScalar>(_b, bcs, _x0, scale);
    VecRestoreArrayRead(x0_local, &array);
    VecGhostRestoreLocalForm(x0, &x0_local);
  }
  else
    fem::set_bc<PetscScalar>(_b, bcs, scale);

  VecRestoreArray(b, &array);
}
//-----------------------------------------------------------------------------
