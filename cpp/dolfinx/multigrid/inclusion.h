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
#include <vector>

#include <mpi.h>

#include "dolfinx/common/IndexMap.h"
#include "dolfinx/la/SparsityPattern.h"
#include "dolfinx/mesh/Mesh.h"

namespace dolfinx::multigrid
{

template <std::floating_point T>
std::vector<std::int64_t>
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

  std::vector<std::int64_t> map(im_from.size_global(), -1);

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

  for (std::int32_t i = 0; i < im_from.size_local(); i++)
  {
    std::ranges::subrange vertex_from(std::next(x_from.begin(), 3 * i),
                                      std::next(x_from.begin(), 3 * (i + 1)));
    for (std::int64_t j = 0; j < im_to.size_local(); j++) // + im_to.num_ghosts() TODO
    {
      std::ranges::subrange vertex_to(std::next(x_to.begin(), 3 * j),
                                      std::next(x_to.begin(), 3 * (j + 1)));

      if (std::ranges::equal(
              vertex_from, vertex_to, [](T a, T b)
              { return std::abs(a - b) <= std::numeric_limits<T>::epsilon(); }))
      {
        map[to_global_from(i)] = to_global_to(j);
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

  // map holds at this point for every original local index the corresponding
  // mapped global index (if it was available on the same process on the to
  // mesh).
  std::vector<std::int64_t> result(map.size(), -1);
  MPI_Allreduce(map.data(), result.data(), map.size(),
                dolfinx::MPI::mpi_t<std::int64_t>, MPI_MAX, mesh_from.comm());

  if (std::ranges::all_of(result, [](auto e) { return e >= 0; }))
    // All vertices indentified
    return result;

  // Build global to vertex list

  // 1) exchange local sizes
  std::vector<std::int32_t> local_sizes(dolfinx::MPI::size(mesh_from.comm()));
  {
    std::array<std::int32_t, 1> tmp{im_to.size_local() * 3};
    MPI_Allgather(&tmp, 1, MPI_INT32_T, local_sizes.data(), 1, MPI_INT32_T,
                  mesh_from.comm());
  }

  // for (auto ls : local_sizes)
  //   std::cout << ls << ", ";

  // 2) compute displacement vector
  std::vector<std::int32_t> displacements(local_sizes.size() + 1, 0);
  std::partial_sum(local_sizes.begin(), local_sizes.end(),
                   displacements.begin() + 1);

  // for (auto ls : displacements)
  //   std::cout << ls << ", ";

  // 3) Allgather global x vector
  std::vector<T> global_x_to(im_to.size_global() * 3);
  MPI_Allgatherv(mesh_to.geometry().x().data(), im_to.size_local() * 3,
                 dolfinx::MPI::mpi_t<T>, global_x_to.data(), local_sizes.data(),
                 displacements.data(), dolfinx::MPI::mpi_t<T>,
                 mesh_from.comm());

  // Recheck indices on global data structure
  for (std::int32_t i = 0; i < im_from.size_local(); i++)
  {
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
        map[to_global_from(i)] = j;
        break;
      }
    }
  }

  MPI_Allreduce(map.data(), result.data(), map.size(),
                dolfinx::MPI::mpi_t<std::int64_t>, MPI_MAX, mesh_from.comm());

  assert(std::ranges::all_of(result, [&](auto e)
                             { return e >= 0 && e < im_to.size_global(); }));
  return result;
}

} // namespace dolfinx::multigrid
