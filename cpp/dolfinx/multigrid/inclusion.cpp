// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "inclusion.h"

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

template std::vector<std::int32_t>
inclusion_mapping<float>(const dolfinx::mesh::Mesh<float>& mesh_from,
                         const dolfinx::mesh::Mesh<float>& mesh_to);

template std::vector<std::int32_t>
inclusion_mapping<double>(const dolfinx::mesh::Mesh<double>& mesh_from,
                          const dolfinx::mesh::Mesh<double>& mesh_to);

} // namespace dolfinx::multigrid
