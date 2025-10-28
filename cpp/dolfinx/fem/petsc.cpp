// Copyright (C) 2018-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_PETSC

#include "petsc.h"
#include <dolfinx/common/IndexMap.h>
#include <functional>
#include <petscistypes.h>

using namespace dolfinx;

//-----------------------------------------------------------------------------
Vec fem::petsc::create_vector_block(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps)
{
  // FIXME: handle constant block size > 1

  auto [rank_offset, local_offset, ghosts_new, ghost_new_owners]
      = common::stack_index_maps(maps);

  std::vector<std::int64_t> ghosts;
  for (auto& sub_ghost : ghosts_new)
    ghosts.insert(ghosts.end(), sub_ghost.begin(), sub_ghost.end());

  std::vector<int> ghost_owners;
  for (auto& sub_owner : ghost_new_owners)
    ghost_owners.insert(ghost_owners.end(), sub_owner.begin(), sub_owner.end());

  // Create map for combined problem, and create vector
  common::IndexMap index_map(maps[0].first.get().comm(), local_offset.back(),
                             ghosts, ghost_owners);

  // NOTE: Calling
  //
  //     la::petsc::create_vector(const common::IndexMap& map, int bs)
  //
  // can lead to memory errors because the MPI communicator associated
  // with map gets destroyed upon exit from this function. The lifetime
  // of the communicators should be managed correctly, see thread at
  // https://lists.mcs.anl.gov/pipermail/petsc-users/2021-July/044103.html.

  // Get PETSc Vec
  auto range = index_map.local_range();
  assert(range[1] >= range[0]);
  std::int32_t local_size = range[1] - range[0];
  std::vector<PetscInt> _ghosts(ghosts.begin(), ghosts.end());
  Vec x = nullptr;
  PetscErrorCode ierr
      = VecCreateGhost(maps[0].first.get().comm(), local_size, PETSC_DETERMINE,
                       _ghosts.size(), _ghosts.data(), &x);
  if (ierr != 0)
    throw std::runtime_error("Call to PETSc VecCreateGhost failed.");

  return x;
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
  VecCreateNest(vecs.front()->comm(), petsc_vecs.size(), nullptr,
                petsc_vecs.data(), &y);
  return y;
}
//-----------------------------------------------------------------------------
#endif
