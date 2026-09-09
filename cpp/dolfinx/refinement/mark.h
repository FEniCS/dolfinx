// Copyright (C) 2026 Paul T. Kühner
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
#include <dolfinx/la/Vector.h>
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

/// @brief Return local indices of a vector whose value exceeds a fraction of
/// the global maximum value.
///
/// Computes the maximum `max` of @p v over the locally owned entries on every
/// rank of `v.index_map()->comm()`, and returns the local indices `i`,
/// satisfying `v[i] > θ max`. This is commonly referred to as 'maximum
/// marking' in the adaptive finite element literature.
///
/// @pre @p v has block size 1.
/// @pre Ghost entries of @p v are up to date, i.e. `scatter_forward` has
/// been called since the owned entries were last modified.
///
/// @note θ = 1 marks nothing, since no entry can strictly exceed the true
/// maximum. θ = 0 is rejected, since `threshold` would be 0 and the
/// criterion would degenerate to marking every entry with a positive
/// value.
/// @note The threshold is bitwise identical on every rank, so an entry is
/// marked consistently by its owner and by every rank ghosting it.
///
/// @warning Returned indices index @p v, not mesh entities. A DOF index is
/// not an entity index in general (e.g. a reordered DG0 dofmap), even with
/// one DOF per entity. To get entity indices, build @p v directly over the
/// entity's `common::IndexMap` (e.g. `mesh::Topology::index_map`) rather
/// than a dofmap's, or map DOFs to entities via the dofmap yourself.
///
/// @param[in] v Vector of indicators, often with each entry associated with a
///   mesh entity.
/// @param[in] theta Cut-off parameter, 0 < θ ≤ 1.
/// @return Local indices, ascending and including ghosts, of `v`
/// that satisfy `v[i] > θ max`.
template <std::floating_point T>
std::vector<std::int32_t> mark_maximum(const la::Vector<T>& v,
                                       std::type_identity_t<T> theta)
{
  // Validate before the collective below
  if ((theta <= 0) or (theta > 1))
  {
    throw std::invalid_argument(
        std::format("theta must satisfy 0 < theta <= 1, got {}.", theta));
  }

  if (v.bs() != 1)
  {
    throw std::invalid_argument(
        std::format("v must have a block size of 1, got {}.", v.bs()));
  }

  const common::IndexMap& im = *v.index_map();
  std::span<const T> values(v.array());

  // If no local entries, assign a large negative value for local maximum
  const T local_max = im.size_local() == 0
                          ? std::numeric_limits<T>::lowest()
                          : std::ranges::max(values.first(im.size_local()));

  T max = 0;
  MPI_Allreduce(&local_max, &max, 1, dolfinx::MPI::mpi_t<T>, MPI_MAX,
                im.comm());

  const T threshold = theta * max;
  const std::int32_t n = values.size();

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

  spdlog::info("Marking (max) {} / {} (owned + ghost) entries.", indices.size(),
               n);

  return indices;
}

} // namespace dolfinx::refinement
