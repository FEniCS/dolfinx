// Copyright (C) 2018-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

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
