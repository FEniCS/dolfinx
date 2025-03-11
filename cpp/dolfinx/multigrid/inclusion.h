// Copyright (C) 2024 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include "dolfinx/common/IndexMap.h"
#include "dolfinx/la/SparsityPattern.h"
#include "dolfinx/mesh/Mesh.h"

namespace dolfinx::multigrid
{

/**
 * @brief Gathers a global vector from combination of local data.
 * @note Performs an all-to-all communication.
 *
 * @param local local data
 * @param global_size number of global data entries
 * @param comm MPI communicator
 * @return std::vector<T> on communicator gathered global data.
 */
template <std::floating_point T>
std::vector<T> gather_global(std::span<const T> local, std::int64_t global_size,
                             MPI_Comm comm)
{
  // 1) exchange local sizes
  std::vector<std::int32_t> local_sizes(dolfinx::MPI::size(comm));
  {
    std::array<std::int32_t, 1> tmp{local.size()};
    MPI_Allgather(&tmp, 1, MPI_INT32_T, local_sizes.data(), 1, MPI_INT32_T,
                  comm);
  }

  // 2) compute displacement vector
  std::vector<std::int32_t> displacements(local_sizes.size() + 1, 0);
  std::partial_sum(local_sizes.begin(), local_sizes.end(),
                   displacements.begin() + 1);

  // 3) Allgather global vector
  std::vector<T> global(global_size);
  MPI_Allgatherv(local.data(), local.size(), dolfinx::MPI::mpi_t<T>,
                 global.data(), local_sizes.data(), displacements.data(),
                 dolfinx::MPI::mpi_t<T>, comm);

  return global;
}

/**
 * @brief Computes an inclusion map, i.e. local list of global vertex indices of
 * another mesh, between to meshes.
 *
 *
 * @param mesh_from Coarser mesh (domain of the inclusion map)
 * @param mesh_to Finer mesh (range of the inclusion map)
 * @param allow_all_to_all If the vertices of `mesh_from` are not equally
 * spatially parallelized as `mesh_to` an all-to-all gathering of all vertices
 * in `mesh_to` is performed. If true, performs all-to-all gathering, otherwise
 * throws an exception if this becomes necessary.
 * @return std::vector<std::int64_t> Map from local vertex index in `mesh_from`
 * to global vertex index in `mesh_to`, i.e. `mesh_from.geometry.x()[i:i+3] ==
 * mesh_to.geometry.x()[map[i]:map[i]+3]`.
 */
template <std::floating_point T>
std::vector<std::int64_t>
inclusion_mapping(const dolfinx::mesh::Mesh<T>& mesh_from,
                  const dolfinx::mesh::Mesh<T>& mesh_to,
                  bool allow_all_to_all = false)
{
  {
    // Check comms equal
    int result;
    MPI_Comm_compare(mesh_from.comm(), mesh_to.comm(), &result);
    assert(result == MPI_CONGRUENT);
  }

  const common::IndexMap& im_from = *mesh_from.topology()->index_map(0);
  const common::IndexMap& im_to = *mesh_to.topology()->index_map(0);

  std::vector<std::int64_t> map(im_from.size_local() + im_from.num_ghosts(),
                                -1);

  std::span<const T> x_from = mesh_from.geometry().x();
  std::span<const T> x_to = mesh_to.geometry().x();

  auto build_global_to_local = [&](const auto& im)
  {
    return [&](std::int32_t idx)
    {
      std::array<std::int64_t, 1> tmp;
      im.local_to_global(std::vector<std::int32_t>{idx}, tmp);
      return tmp[0];
    };
  };

  auto to_global_to = build_global_to_local(im_to);
  auto to_global_from = build_global_to_local(im_from);

  for (std::int32_t i = 0; i < im_from.size_local() + im_from.num_ghosts(); i++)
  {
    std::ranges::subrange vertex_from(std::next(x_from.begin(), 3 * i),
                                      std::next(x_from.begin(), 3 * (i + 1)));
    for (std::int64_t j = 0; j < im_to.size_local() + im_to.num_ghosts(); j++)
    {
      std::ranges::subrange vertex_to(std::next(x_to.begin(), 3 * j),
                                      std::next(x_to.begin(), 3 * (j + 1)));

      if (std::ranges::equal(
              vertex_from, vertex_to, [](T a, T b)
              { return std::abs(a - b) <= std::numeric_limits<T>::epsilon(); }))
      {
        assert(map[i] == -1);
        map[i] = to_global_to(j);
        break;
      }
    }
  }

  if (dolfinx::MPI::size(mesh_to.comm()) == 1)
  {
    // no communication required
    assert(std::ranges::all_of(map, [](auto e) { return e >= 0; }));
    return map;
  }

  bool all_found = std::ranges::all_of(map, [](auto e) { return e >= 0; });
  MPI_Allreduce(MPI_IN_PLACE, &all_found, 1, MPI_CXX_BOOL, MPI_LAND,
                mesh_to.comm());
  if (all_found)
    return map;

  if (!allow_all_to_all)
  {
    throw std::runtime_error(
        "Parallelization of mesh requires all to all communication to compute "
        "inclusion map, but allow_all_to_all is not set.");
  }

  // Build global to vertex list
  std::vector<T> global_x_to
      = gather_global(mesh_to.geometry().x().subspan(0, im_to.size_local() * 3),
                      im_to.size_global() * 3, mesh_to.comm());

  // Recheck indices on global data structure
  for (std::int32_t i = 0; i < im_from.size_local() + im_from.num_ghosts(); i++)
  {
    // TODO:
    // if (map[i] >= 0)
    //   continue;

    std::ranges::subrange vertex_from(std::next(x_from.begin(), 3 * i),
                                      std::next(x_from.begin(), 3 * (i + 1)));
    for (std::int64_t j = 0; j < im_to.size_global(); j++)
    {
      std::ranges::subrange vertex_to(
          std::next(global_x_to.begin(), 3 * j),
          std::next(global_x_to.begin(), 3 * (j + 1)));

      if (std::ranges::equal(
              vertex_from, vertex_to, [](T a, T b)
              { return std::abs(a - b) <= std::numeric_limits<T>::epsilon(); }))
      {
        map[i] = j;
        break;
      }
    }
  }

  assert(std::ranges::all_of(map, [&](auto e)
                             { return e >= 0 && e < im_to.size_global(); }));
  return map;
}

} // namespace dolfinx::multigrid
