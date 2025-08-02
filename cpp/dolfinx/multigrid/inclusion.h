// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <vector>

#include <mpi.h>

#include "dolfinx/common/IndexMap.h"
#include "dolfinx/la/SparsityPattern.h"
#include "dolfinx/mesh/Mesh.h"

namespace dolfinx::multigrid
{

/**
 * @brief Computes an inclusion map: a map between vertex indices from one mesh
 * to another.
 *
 * @param mesh_from Domain of the map
 * @param mesh_to Range of the map
 *
 * @return Inclusion map, the `i`-th component is the vertex index of the vertex
 * with the same coordinates in `mesh_to` and `-1` if it can not be found
 * (locally!) in `mesh_to`. If `map[i] != -1` it holds
 * `mesh_from.geometry.x()[i:i+3] == mesh_to.geometry.x()[map[i]:map[i]+3]`.
 *
 * @note Invoking `inclusion_map` on a `(mesh_coarse, mesh_fine)` tuple, where
 * `mesh_fine` is produced by refinement with `IdentityPartitionerPlaceholder()`
 * option, the returned `map` is guaranteed to match all vertices for all
 * locally owned vertices (not for the ghost vertices).
 */
template <std::floating_point T>
std::vector<std::int32_t>
inclusion_mapping(const dolfinx::mesh::Mesh<T>& mesh_from,
                  const dolfinx::mesh::Mesh<T>& mesh_to)
{
  {
    // Check comms equal
    int result;
    MPI_Comm_compare(mesh_from.comm(), mesh_to.comm(), &result);
    assert(result == MPI_CONGRUENT);
  }

  const common::IndexMap& im_from = *mesh_from.topology()->index_map(0);
  const common::IndexMap& im_to = *mesh_to.topology()->index_map(0);

  std::vector<std::int32_t> map(im_from.size_local() + im_from.num_ghosts(),
                                -1);

  std::span<const T> x_from = mesh_from.geometry().x();
  std::span<const T> x_to = mesh_to.geometry().x();

  for (std::int32_t i = 0; i < im_from.size_local() + im_from.num_ghosts(); i++)
  {
    std::ranges::subrange vertex_from(std::next(x_from.begin(), 3 * i),
                                      std::next(x_from.begin(), 3 * (i + 1)));
    for (std::int32_t j = 0; j < im_to.size_local() + im_to.num_ghosts(); j++)
    {
      std::ranges::subrange vertex_to(std::next(x_to.begin(), 3 * j),
                                      std::next(x_to.begin(), 3 * (j + 1)));

      if (std::ranges::equal(
              vertex_from, vertex_to, [](auto a, auto b)
              { return std::abs(a - b) <= std::numeric_limits<T>::epsilon(); }))
      {
        assert(map[i] == -1);
        map[i] = j;
        break;
      }
    }
  }

  return map;
}

} // namespace dolfinx::multigrid
