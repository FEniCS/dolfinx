// Copyright (C) 2018-2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "petsc.h"
#include "SparsityPatternBuilder.h"
#include "assembler.h"
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/la/SparsityPattern.h>

using namespace dolfinx;

//-----------------------------------------------------------------------------
la::PETScMatrix dolfinx::fem::create_matrix(const Form<PetscScalar>& a)
{
  // Build sparsitypattern
  la::SparsityPattern pattern = fem::create_sparsity_pattern(a);

  // Finalise communication
  pattern.assemble();

  // Initialize matrix
  common::Timer t1("Init tensor");
  la::PETScMatrix A(a.mesh()->mpi_comm(), pattern);
  t1.stop();

  return A;
}
//-----------------------------------------------------------------------------
la::PETScMatrix fem::create_matrix_block(
    const Eigen::Ref<
        const Eigen::Array<const fem::Form<PetscScalar>*, Eigen::Dynamic,
                           Eigen::Dynamic, Eigen::RowMajor>>& a)
{
  // Extract and check row/column ranges
  auto V = function::common_function_spaces(extract_function_spaces(a));

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
      if (a(row, col))
      {
        // Create sparsity pattern for block
        patterns[row].push_back(std::make_unique<la::SparsityPattern>(
            mesh->mpi_comm(), index_maps));

        // Build sparsity pattern for block
        std::array dofmaps{V[0][row]->dofmap().get(),
                           V[1][col]->dofmap().get()};
        assert(patterns[row].back());
        auto& sp = patterns[row].back();
        assert(sp);
        const FormIntegrals<PetscScalar>& integrals = a(row, col)->integrals();
        if (integrals.num_integrals(IntegralType::cell) > 0)
          SparsityPatternBuilder::cells(*sp, mesh->topology(), dofmaps);
        if (integrals.num_integrals(IntegralType::interior_facet) > 0)
        {
          mesh->topology_mutable().create_entities(tdim - 1);
          SparsityPatternBuilder::interior_facets(*sp, mesh->topology(),
                                                  dofmaps);
        }
        if (integrals.num_integrals(IntegralType::exterior_facet) > 0)
        {
          mesh->topology_mutable().create_entities(tdim - 1);
          SparsityPatternBuilder::exterior_facets(*sp, mesh->topology(),
                                                  dofmaps);
        }
      }
      else
        patterns[row].push_back(nullptr);
    }
  }

  // Compute offsets for the fields
  std::array<std::vector<std::reference_wrapper<const common::IndexMap>>, 2>
      maps;
  for (std::size_t d = 0; d < 2; ++d)
  {
    for (auto space : V[d])
      maps[d].push_back(*space->dofmap()->index_map.get());
  }

  // FIXME: This is computed again inside the SparsityPattern
  // constructor, but we also need to outside to build the PETSc
  // local-to-global map. Compute outside and pass into SparsityPattern
  // constructor.
  auto [rank_offset, local_offset, ghosts, owner]
      = common::stack_index_maps(maps[0]);

  // Create merged sparsity pattern
  std::vector<std::vector<const la::SparsityPattern*>> p(V[0].size());
  for (std::size_t row = 0; row < V[0].size(); ++row)
    for (std::size_t col = 0; col < V[1].size(); ++col)
      p[row].push_back(patterns[row][col].get());
  la::SparsityPattern pattern(mesh->mpi_comm(), p, maps);
  pattern.assemble();

  // FIXME: Add option to pass customised local-to-global map to PETSc
  // Mat constructor.

  // Initialise matrix
  la::PETScMatrix A(mesh->mpi_comm(), pattern);

  // Create row and column local-to-global maps (field0, field1, field2,
  // etc), i.e. ghosts of field0 appear before owned indices of field1
  std::array<std::vector<PetscInt>, 2> _maps;
  for (int d = 0; d < 2; ++d)
  {
    for (std::size_t f = 0; f < maps[d].size(); ++f)
    {
      const common::IndexMap& map = maps[d][f].get();
      const int bs = map.block_size();
      const std::int32_t size_local = bs * map.size_local();
      const std::vector global = map.global_indices(false);
      for (std::int32_t i = 0; i < size_local; ++i)
        _maps[d].push_back(i + rank_offset + local_offset[f]);
      for (std::size_t i = size_local; i < global.size(); ++i)
        _maps[d].push_back(ghosts[f][i - size_local]);
    }
  }

  // Create PETSc local-to-global map/index sets and attach to matrix
  ISLocalToGlobalMapping petsc_local_to_global0, petsc_local_to_global1;
  ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 1, _maps[0].size(),
                               _maps[0].data(), PETSC_COPY_VALUES,
                               &petsc_local_to_global0);
  ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 1, _maps[1].size(),
                               _maps[1].data(), PETSC_COPY_VALUES,
                               &petsc_local_to_global1);
  MatSetLocalToGlobalMapping(A.mat(), petsc_local_to_global0,
                             petsc_local_to_global1);

  // Clean up local-to-global maps
  ISLocalToGlobalMappingDestroy(&petsc_local_to_global0);
  ISLocalToGlobalMappingDestroy(&petsc_local_to_global1);

  return A;
}
//-----------------------------------------------------------------------------
la::PETScMatrix fem::create_matrix_nest(
    const Eigen::Ref<
        const Eigen::Array<const fem::Form<PetscScalar>*, Eigen::Dynamic,
                           Eigen::Dynamic, Eigen::RowMajor>>& a)
{
  // Extract and check row/column ranges
  auto V = function::common_function_spaces(extract_function_spaces(a));

  // Loop over each form and create matrix
  Eigen::Array<std::shared_ptr<la::PETScMatrix>, Eigen::Dynamic, Eigen::Dynamic,
               Eigen::RowMajor>
      mats(a.rows(), a.cols());
  Eigen::Array<Mat, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> petsc_mats(
      a.rows(), a.cols());
  for (int i = 0; i < a.rows(); ++i)
  {
    for (int j = 0; j < a.cols(); ++j)
    {
      if (a(i, j))
      {
        mats(i, j) = std::make_shared<la::PETScMatrix>(create_matrix(*a(i, j)));
        petsc_mats(i, j) = mats(i, j)->mat();
      }
      else
        petsc_mats(i, j) = nullptr;
    }
  }

  // Initialise block (MatNest) matrix
  Mat _A;
  MatCreate(V[0][0]->mesh()->mpi_comm(), &_A);
  MatSetType(_A, MATNEST);
  MatNestSetSubMats(_A, petsc_mats.rows(), nullptr, petsc_mats.cols(), nullptr,
                    petsc_mats.data());
  MatSetUp(_A);

  return la::PETScMatrix(_A);
}
//-----------------------------------------------------------------------------
la::PETScVector fem::create_vector_block(
    const std::vector<std::reference_wrapper<const common::IndexMap>>& maps)
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

  std::vector<int> dest_ranks;
  for (auto& map : maps)
  {
    const auto [_, ranks] = dolfinx::MPI::neighbors(
        map.get().comm(common::IndexMap::Direction::forward));
    dest_ranks.insert(dest_ranks.end(), ranks.begin(), ranks.end());
  }
  std::sort(dest_ranks.begin(), dest_ranks.end());
  dest_ranks.erase(std::unique(dest_ranks.begin(), dest_ranks.end()),
                   dest_ranks.end());

  // Create map for combined problem, and create vector
  common::IndexMap index_map(maps[0].get().comm(), local_size, dest_ranks,
                             ghosts, ghost_owners, 1);

  return la::PETScVector(index_map);
}
//-----------------------------------------------------------------------------
la::PETScVector
fem::create_vector_nest(const std::vector<const common::IndexMap*>& maps)
{
  assert(!maps.empty());

  // Loop over each form and create vector
  std::vector<std::shared_ptr<la::PETScVector>> vecs(maps.size());
  std::vector<Vec> petsc_vecs(maps.size());
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    if (!maps[i])
    {
      throw std::runtime_error(
          "Cannot construct nested PETSc vectors with null blocks.");
    }
    vecs[i] = std::make_shared<la::PETScVector>(*maps[i]);
    petsc_vecs[i] = vecs[i]->vec();
  }

  // Create nested (VecNest) vector
  Vec y;
  VecCreateNest(vecs[0]->mpi_comm(), petsc_vecs.size(), nullptr,
                petsc_vecs.data(), &y);
  return la::PETScVector(y, false);
}
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
  fem::assemble_vector<PetscScalar>(_b, L);
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
    fem::apply_lifting<PetscScalar>(_b, a, bcs1, {}, scale);
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
    fem::set_bc<PetscScalar>(_b, bcs, _x0, scale);
    VecRestoreArrayRead(x0_local, &array);
    VecGhostRestoreLocalForm(x0, &x0_local);
  }
  else
    fem::set_bc<PetscScalar>(_b, bcs, scale);

  VecRestoreArray(b, &array);
}
//-----------------------------------------------------------------------------
