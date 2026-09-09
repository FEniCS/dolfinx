// Copyright (C) 2026 Paul T. Kühner and Jack S. Hale
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <format>
#include <limits>
#include <mpi.h>
#include <span>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace dolfinx::refinement
{

/// @brief Return local indices of a set of values whose entry exceeds a
/// fraction of the global maximum value.
///
/// Computes the maximum `max` of @p values over the locally owned entries on
/// every rank of `index_map.comm()`, and returns the local indices `i`,
/// satisfying `values[i] > θ max`. This is commonly referred to as 'maximum
/// marking' in the adaptive finite element literature.
///
/// @pre @p values has size `index_map.size_local() + index_map.num_ghosts()`.
/// @pre Ghost entries of @p values are up to date, i.e. `scatter_forward` has
/// been called since the owned entries were last modified.
///
/// @note θ = 1 marks nothing, since no entry can strictly exceed the true
/// maximum. θ = 0 is rejected, since `threshold` would be 0 and the
/// criterion would degenerate to marking every entry with a positive
/// value.
/// @note The threshold is bitwise identical on every rank, so an entry is
/// marked consistently by its owner and by every rank ghosting it.
///
/// @warning Returned indices index @p values, not mesh entities. A DOF index
/// is not an entity index in general (e.g. a reordered DG0 dofmap), even with
/// one DOF per entity. To get entity indices, build @p values and
/// @p index_map directly from the entity's `common::IndexMap` (e.g.
/// `mesh::Topology::index_map`) rather than a dofmap's, or map DOFs to
/// entities via the dofmap yourself.
///
/// @param[in] values Values, often with each entry associated with a mesh
///   entity, e.g. an error indicator.
/// @param[in] index_map Index map describing the parallel layout of @p
///   values.
/// @param[in] theta Cut-off parameter, 0 < θ ≤ 1.
/// @return Local indices, ascending and including ghosts, of `values`
/// that satisfy `values[i] > θ max`.
template <std::floating_point T>
std::vector<std::int32_t> mark_maximum(std::span<const T> values,
                                       const common::IndexMap& index_map,
                                       std::type_identity_t<T> theta)
{
  // Validate before the collective below
  if ((theta <= 0) or (theta > 1))
  {
    throw std::invalid_argument(
        std::format("theta must satisfy 0 < theta <= 1, got {}.", theta));
  }

  const std::int32_t n = values.size();
  const std::int32_t size = index_map.size_local() + index_map.num_ghosts();
  if (n != size)
  {
    throw std::invalid_argument(
        std::format("values must have size index_map.size_local() + "
                    "index_map.num_ghosts() = {}, got {}.",
                    size, n));
  }

  // If no local entries, assign a large negative value for local maximum
  const T local_max
      = index_map.size_local() == 0
            ? std::numeric_limits<T>::lowest()
            : std::ranges::max(values.first(index_map.size_local()));

  T max = 0;
  MPI_Allreduce(&local_max, &max, 1, dolfinx::MPI::mpi_t<T>, MPI_MAX,
                index_map.comm());

  const T threshold = theta * max;

  auto mark = [threshold](T e) { return e > threshold; };

  std::vector<std::int32_t> indices;
  indices.reserve(std::ranges::count_if(values, mark));
  // TODO: replace with std::views::enumerate(values) once dolfinx targets
  // C++23
  // values includes ghosts
  for (std::int32_t i = 0; i < n; ++i)
  {
    if (mark(values[i]))
      indices.push_back(i);
  }

  spdlog::info("Marking (maximum): marked {} of {} local entries (owned + "
               "ghost).",
               indices.size(), n);

  return indices;
}

} // namespace dolfinx::refinement
